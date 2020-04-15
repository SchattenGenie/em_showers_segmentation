import comet_ml
# from comet_ml import OfflineExperiment as Experiment
from comet_ml import Experiment as Experiment
import click
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data._utils import pin_memory
from nets import GraphNN_KNN_v1, EdgeClassifier_v1
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, accuracy_score, average_precision_score
from torch_geometric.data import DataLoader
from preprocessing import preprocess_dataset
from viz_utils import RunningAverageMeter, plot_aucs
import matplotlib.pyplot as plt
from tqdm import tqdm
import classification_losses
from sys_utils import get_freer_gpu
from roc_auc.roc_auc import fast_auc
from pathlib import Path
import sys
import os


def str_to_class(classname: str):
    """
    Function to get class object by its name signature
    :param classname: str
        name of the class
    :return: class object with the same name signature as classname
    """
    return getattr(sys.modules[__name__], classname)


def predict_one_shower(shower, graph_embedder, edge_classifier):
    # TODO: batch training
    embeddings = graph_embedder(shower)
    edge_labels_true = (shower.y[shower.edge_index[0]] == shower.y[shower.edge_index[1]]).view(-1)
    edge_labels_predicted = edge_classifier(shower=shower, embeddings=embeddings, edge_index=shower.edge_index).view(-1)
    return edge_labels_true, torch.clamp(edge_labels_predicted, 1e-6, 1 - 1e-6)


@click.command()
@click.option('--datafile', type=str, default='./data/train_200_preprocessed.pt')
@click.option('--project_name', type=str, prompt='Enter project name', default='em_showers_network_training')
@click.option('--workspace', type=str, prompt='Enter workspace name')
@click.option('--epochs', type=int, default=4000)
@click.option('--learning_rate', type=float, default=1e-3)
@click.option('--hidden_dim', type=int, default=32)
@click.option('--output_dim', type=int, default=32)
@click.option('--num_layers_emulsion', type=int, default=3)
@click.option('--num_layers_edge_conv', type=int, default=5)
@click.option('--graph_embedder', type=str, default='GraphNN_KNN_v1')
@click.option('--edge_classifier', type=str, default='EdgeClassifier_v1')
@click.option('--outer_optimization', type=bool, default=False)
@click.option('--optimizer', type=str, default='Adam')
@click.option('--use_scheduler', type=bool, default=False)
def main(
        datafile='./data/train_200_preprocessed.pt',
        epochs=1000,
        learning_rate=1e-3,
        hidden_dim=12,
        output_dim=12,
        num_layers_emulsion=3,
        num_layers_edge_conv=3,
        project_name='em_showers_net_training',
        workspace='schattengenie',
        graph_embedder='GraphNN_KNN_v1',
        edge_classifier='EdgeClassifier_v1',
        outer_optimization=False,
        optimizer='Adam',
        use_scheduler=False
):
    if torch.cuda.is_available():
        device = torch.device('cuda:0')  # torch.device('cuda:{}'.format(get_freer_gpu()))
    else:
        device = torch.device('cpu')
    print("Using device = {}".format(device))
    # offline_directory="/home/vbelavin/comet_ml_offline")
    experiment = Experiment(project_name=project_name, workspace=workspace)
    device = torch.device(device)
    showers = torch.load(datafile)
    showers_train, showers_test = train_test_split(showers, random_state=1337)

    train_loader = DataLoader(showers_train, batch_size=1, shuffle=True)
    test_loader = DataLoader(showers_test, batch_size=1, shuffle=True)

    input_dim = showers[0].x.shape[1]
    edge_dim = showers[0].edge_features.shape[1]
    graph_embedder = str_to_class(graph_embedder)(
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        edge_dim=edge_dim,
        num_layers_emulsion=num_layers_emulsion,
        num_layers_edge_conv=num_layers_edge_conv,
        input_dim=input_dim,
    ).to(device)
    edge_classifier = str_to_class(edge_classifier)(
        input_dim=2 * output_dim + edge_dim,
    ).to(device)
    optimizer = getattr(torch.optim, optimizer)(
        list(graph_embedder.parameters()) + list(edge_classifier.parameters()),
        lr=learning_rate
    )
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=100)

    class_prior = 0.025
    criterion = classification_losses.FocalLoss(gamma=3., alpha=1 - class_prior, device=device).to(device)

    loss_train = RunningAverageMeter()
    loss_test = RunningAverageMeter()
    roc_auc_test = RunningAverageMeter()
    acc_test = RunningAverageMeter()
    class_disbalance = RunningAverageMeter()
    best_roc_auc = 0.
    for epoch in tqdm(range(epochs)):
        if outer_optimization: optimizer.zero_grad()
        for shower in train_loader:
            if not outer_optimization: optimizer.zero_grad()
            shower = shower.to(device)
            edge_labels_true, edge_labels_predicted = predict_one_shower(shower,
                                                                         graph_embedder=graph_embedder,
                                                                         edge_classifier=edge_classifier)
            # calculate the batch loss
            loss = criterion(edge_labels_predicted, edge_labels_true.float())
            # Zero gradients, perform a backward pass, and update the weights.
            if not outer_optimization:
                loss.backward()
            else:
                (loss / len(train_loader)).backward()
            loss_train.update(loss.item())
            class_disbalance.update((edge_labels_true.sum().float() / len(edge_labels_true)).item())
            if not outer_optimization: optimizer.step()
        if outer_optimization: optimizer.step()
        torch.cuda.empty_cache()
        y_true_list = []
        y_pred_list = []
        for shower in test_loader:
            shower = shower.to(device)
            with torch.no_grad():
                edge_labels_true, edge_labels_predicted = predict_one_shower(shower,
                                                                             graph_embedder=graph_embedder,
                                                                             edge_classifier=edge_classifier)

                # calculate the batch loss
                loss = criterion(edge_labels_predicted, edge_labels_true.float())
            y_true, y_pred = edge_labels_true.detach().cpu().numpy(), edge_labels_predicted.detach().cpu().numpy()
            y_true_list.append(y_true)
            y_pred_list.append(y_pred)
            acc = accuracy_score(y_true, y_pred.round())
            roc_auc = fast_auc(y_true.astype(np.float64), y_pred.astype(np.float64)) - fast_auc(y_true.astype(np.float64), -shower.edge_attr.view(-1).detach().cpu().numpy().astype(np.float64))
            loss_test.update(loss.item())
            acc_test.update(acc)
            roc_auc_test.update(roc_auc)
            class_disbalance.update((edge_labels_true.sum().float() / len(edge_labels_true)).item())
        torch.cuda.empty_cache()
        print("loss_test.val={}, roc_auc_test.val={}".format(loss_test.val, roc_auc_test.val))
        if use_scheduler:
            scheduler.step(roc_auc_test.val)

        experiment.log_metric('loss_test', loss_test.val)
        experiment.log_metric('acc_test', acc_test.val)
        experiment.log_metric('roc_auc_test', roc_auc_test.val)
        experiment.log_metric('class_disbalance', class_disbalance.val)

        # y_true = np.concatenate(y_true_list)
        # y_pred = np.concatenate(y_pred_list)
        # f = plot_aucs(y_true=y_true, y_pred=y_pred)
        # experiment.log_figure("Optimization dynamic", f, overwrite=True)
        # plt.close(f)
        experiment_key = experiment.get_key()
        if roc_auc_test.val > best_roc_auc:
            best_roc_auc = roc_auc_test.val
            torch.save(graph_embedder.state_dict(), "graph_embedder_{}_{}.pt".format(Path(datafile).stem, experiment_key))
            torch.save(edge_classifier.state_dict(), "edge_classifier_{}_{}.pt".format(Path(datafile).stem, experiment_key))


if __name__ == "__main__":
    main()
