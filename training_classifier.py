import comet_ml
from comet_ml import Experiment
import click
import numpy as np
import torch
import torch.nn as nn
from nets import *
from hgcn.models.hgcn_nets import HGCN, HGCNResidual, HGCNResidualEmulsionConv
from hgcn.optimizers import RiemannianAdam
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, accuracy_score, average_precision_score
from torch_geometric.data import DataLoader
from preprocessing import preprocess_dataset
from viz_utils import RunningAverageMeter, plot_aucs
from hgcn import manifolds
from tqdm import tqdm
import classification_losses
import sys
import os


def get_freer_gpu():
    """
    Function to get the freest GPU available in the system
    :return:
    """
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)


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
    return edge_labels_true, torch.clamp(2 * edge_labels_predicted, 1e-6, 1 - 1e-6)


@click.command()
@click.option('--datafile', type=str, default='./data/train.pt')
@click.option('--project_name', type=str, prompt='Enter project name', default='em_showers_network_training')
@click.option('--work_space', type=str, prompt='Enter workspace name')
@click.option('--epochs', type=int, default=1000)
@click.option('--learning_rate', type=float, default=1e-3)
@click.option('--gcn_dim', type=int, default=12)
@click.option('--hidden_dim', type=int, default=12)
@click.option('--num_layers', type=int, default=3)
@click.option('--hyperbolic', type=bool, default=False)
@click.option('--graph_embedder', type=str, default='GraphNN_KNN_v1')
@click.option('--edge_classifier', type=str, default='EdgeClassifier_v1')
def main(
        datafile='./data/train_.pt',
        epochs=1000,
        learning_rate=1e-3,
        gcn_dim=12,
        hidden_dim=12,  # obsolete for now ;()
        num_layers=3,
        project_name='em_showers_net_training',
        work_space='schattengenie',
        hyperbolic=False,
        graph_embedder='GraphNN_KNN_v1',
        edge_classifier='EdgeClassifier_v1'
):
    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(get_freer_gpu()))
    else:
        device = torch.device('cpu')
    print("Using device = {}".format(device))

    edge_dim = 1
    experiment = Experiment(project_name=project_name, workspace=work_space)
    device = torch.device(device)
    showers = preprocess_dataset(datafile)
    showers_train, showers_test = train_test_split(showers, random_state=1337)

    train_loader = DataLoader(showers_train, batch_size=1, shuffle=True)
    test_loader = DataLoader(showers_test, batch_size=1, shuffle=True)

    input_dim = showers[0].x.shape[1]

    if hyperbolic:
        manifold = manifolds.PoincareBall()
        graph_embedder = str_to_class(graph_embedder)(
            manifold=manifold,
            output_dim=gcn_dim,
            hidden_dim=gcn_dim,
            input_dim=input_dim,
            edge_dim=edge_dim,
            num_layers=num_layers
        ).to(device)

        edge_classifier = str_to_class(edge_classifier)(
            manifold=manifold,
            input_dim=2 * gcn_dim + edge_dim,
            c=graph_embedder.curvatures[-1]
        ).to(device)
        optimizer = RiemannianAdam(
            list(graph_embedder.parameters()) + list(edge_classifier.parameters()),
            lr=learning_rate, stabilize=True
        )
    else:
        graph_embedder = str_to_class(graph_embedder)(
            output_dim=gcn_dim,
            hidden_dim=gcn_dim,
            input_dim=input_dim,
            num_layers=num_layers
        ).to(device)
        edge_classifier = str_to_class(edge_classifier)(
            input_dim=2 * gcn_dim + edge_dim
        ).to(device)
        optimizer = torch.optim.Adam(
            list(graph_embedder.parameters()) + list(edge_classifier.parameters()),
            lr=learning_rate  # , stabilize=True
        )
    # criterion = nn.BCELoss()
    criterion = classification_losses.FocalLoss(gamma=2.)
    # torch.optim.Adam
    loss_train = RunningAverageMeter()
    loss_test = RunningAverageMeter()
    roc_auc_test = RunningAverageMeter()
    pr_auc_test = RunningAverageMeter()
    acc_test = RunningAverageMeter()
    class_disbalance = RunningAverageMeter()

    for _ in tqdm(range(epochs)):
        for shower in train_loader:
            shower = shower.to(device)
            edge_labels_true, edge_labels_predicted = predict_one_shower(shower,
                                                                         graph_embedder=graph_embedder,
                                                                         edge_classifier=edge_classifier)
            # calculate the batch loss
            loss = criterion(edge_labels_predicted, edge_labels_true.float())
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train.update(loss.item())
            class_disbalance.update((edge_labels_true.sum().float() / len(edge_labels_true)).item())

        y_true_list = []
        y_pred_list = []
        for shower in test_loader:
            shower = shower.to(device)
            edge_labels_true, edge_labels_predicted = predict_one_shower(shower,
                                                                         graph_embedder=graph_embedder,
                                                                         edge_classifier=edge_classifier)

            # calculate the batch loss
            loss = criterion(edge_labels_predicted, edge_labels_true.float())
            y_true, y_pred = edge_labels_true.detach().cpu().numpy(), edge_labels_predicted.detach().cpu().numpy()
            y_true_list.append(y_true)
            y_pred_list.append(y_pred)
            acc = accuracy_score(y_true, y_pred.round())
            roc_auc = roc_auc_score(y_true, y_pred)
            pr_auc = average_precision_score(y_true, y_pred)
            loss_test.update(loss.item())
            acc_test.update(acc)
            roc_auc_test.update(roc_auc)
            pr_auc_test.update(pr_auc)
            print(loss.item(), roc_auc, pr_auc)
            class_disbalance.update((edge_labels_true.sum().float() / len(edge_labels_true)).item())


        experiment.log_metric('loss_test', loss_test.val)
        experiment.log_metric('acc_test', acc_test.val)
        experiment.log_metric('roc_auc_test', roc_auc_test.val)
        experiment.log_metric('pr_auc_test', pr_auc_test.val)
        experiment.log_metric('class_disbalance', class_disbalance.val)

        y_true = np.concatenate(y_true_list)
        y_pred = np.concatenate(y_pred_list)
        f = plot_aucs(y_true=y_true, y_pred=y_pred)
        experiment.log_figure("Optimization dynamic", f, overwrite=True)
        experiment_key = experiment.get_key()
        # TODO: save best
        torch.save(graph_embedder.state_dict(), "graph_embedder_{}.pt".format(experiment_key))
        torch.save(edge_classifier.state_dict(), "edge_classifier_{}.pt".format(experiment_key))


if __name__ == "__main__":
    main()
