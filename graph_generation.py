import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import networkx as nx
import uproot
from collections import deque
from tools.opera_distance_metric import generate_k_nearest_graph, opera_distance_metric_py, generate_radius_graph
import torch
import torch_scatter
import torch_geometric
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from collections import defaultdict
from joblib import Parallel, delayed
import click

# !cd tools/ && python setup_opera_distance_metric.py build_ext --inplace


def load_mc(filename="./EM_data/mcdata_taue2.root", step=1):
    f = uproot.open(filename)
    mc = f['Data'].pandas.df(["Event_id", "ele_P", "BT_X", "BT_Y",
                              "BT_Z","BT_SX", "BT_SY","ele_x",
                              "ele_y", "ele_z", "ele_sx", "ele_sy", "chisquare", ], flatten=False)
    pmc = pd.DataFrame(mc)
    pmc['numtracks'] = pmc.BT_X.apply(lambda x: len(x))
    # cuts
    shapechange = [pmc.shape[0]]
    pmc = pmc[pmc.ele_P > 0.1]
    shapechange.append(pmc.shape[0])

    pmc = pmc[pmc.ele_z < 0]
    shapechange.append(pmc.shape[0])

    pmc = pmc[pmc.numtracks > 70]
    shapechange.append(pmc.shape[0])
    print("numtracks reduction by cuts: ", shapechange)
    pmc['m_BT_X'] = pmc.BT_X.apply(lambda x: x.mean())
    pmc['m_BT_Y'] = pmc.BT_Y.apply(lambda x: x.mean())
    pmc['m_BT_Z'] = pmc.BT_Z.apply(lambda x: x.mean())

    print("len(pmc): {len}".format(len=len(pmc)))
    return pmc


def pmc_to_ship_format(pmc, num_showers_in_brick):
    showers = defaultdict(list)
    for i, idx in enumerate(pmc.index):
        shower = pmc.loc[idx]
        n = len(shower['BT_X'])
        showers['SX'].extend(shower['BT_X'])
        showers['SY'].extend(shower['BT_Y'])
        showers['SZ'].extend(shower['BT_Z'])
        showers['TX'].extend(shower['BT_SX'])
        showers['TY'].extend(shower['BT_SY'])

        showers['ele_P'].extend(n * [shower['ele_P']])
        showers['ele_SX'].extend(n * [shower['ele_x']])
        showers['ele_SY'].extend(n * [shower['ele_y']])
        showers['ele_SZ'].extend(n * [shower['ele_z']])
        showers['ele_TX'].extend(n * [shower['ele_sx']])
        showers['ele_TY'].extend(n * [shower['ele_sy']])

        showers['numtracks'].extend(n * [shower['numtracks']])
        showers['signal'].extend(n * [i % num_showers_in_brick])
        showers['brick_id'].extend(n * [i // num_showers_in_brick])

    return showers


def gen_one_shower(df_brick, knn=False, r=250, k=5, symmetric=False, directed=False, e=0.00005, scale=1e4):
    print('Start!')
    from tools.opera_distance_metric import generate_k_nearest_graph, opera_distance_metric_py, generate_radius_graph
    if knn:
        edges_from, edge_to, dist = generate_k_nearest_graph(
            df_brick[["brick_id", "SX", "SY", "SZ", "TX", "TY"]].values,
            k,
            e=e,
            symmetric=symmetric, directed=directed)
        edges = np.vstack([edges_from, edge_to])
        dist = np.array(dist)
        edge_index = torch.LongTensor(edges)
    else:
        edges_from, edge_to, dist = generate_radius_graph(
            df_brick[["brick_id", "SX", "SY", "SZ", "TX", "TY"]].values,
            r,
            e=e,
            symmetric=symmetric, directed=directed)
        edges = np.vstack([edges_from, edge_to])
        dist = np.array(dist)
        edge_index = torch.LongTensor(edges)

    x = torch.FloatTensor(df_brick[["SX", "SY", "SZ", "TX", "TY"]].values / np.array([scale, scale, scale, 1., 1.]))
    shower_data = torch.FloatTensor(
        df_brick[["ele_P", "ele_SX", "ele_SY", "ele_SZ", "ele_TX", "ele_TY", "numtracks", "signal"]].values / np.array(
            [1., scale, scale, scale, 1., 1., 1., 1.]))
    edge_attr = torch.log(torch.FloatTensor(dist).view(-1, 1))
    y = torch.LongTensor(df_brick.signal.values)
    shower = torch_geometric.data.Data(
        x=x,
        edge_index=edge_index,
        shower_data=shower_data,
        pos=x,
        edge_attr=edge_attr,
        y=y
    )
    return shower


def gen_torch_showers(df, knn=False, r=250, k=5, symmetric=False, directed=False, e=0.00005, scale=1e4):
    df_bricks = [df[df.brick_id == brick_id] for brick_id in list(df.brick_id.unique())][:3]
    showers = Parallel(n_jobs=10)(
        delayed(gen_one_shower)(df_brick, knn=knn, r=r, k=k, symmetric=symmetric, directed=directed, e=e, scale=scale) for df_brick in
        df_bricks)
    return showers


@click.command()
@click.option('--root_file', type=str, default='./data/mcdata_taue2.root')
@click.option('--output_file', type=str, default='./data/train.pt')
@click.option('--knn', type=bool, default=True)
@click.option('--k', type=int, default=10)
@click.option('--r', type=int, default=400)
@click.option('--directed', type=bool, default=False)
@click.option('--symmetric', type=bool, default=False)
@click.option('--e', type=float, default=10)
@click.option('--num_showers_in_brick', type=int, default=200)
def main(
        root_file='./data/mcdata_taue2.root',
        output_file='./data/train.pt',
        knn=True,
        k=10,
        r=400,
        directed=False,
        symmetric=False,
        e=10,
        num_showers_in_brick=200
):
    pmc = load_mc(filename=root_file, step=1)
    print(pmc.columns)
    pmc = pmc.loc[(pmc["BT_X"].apply(lambda x: len(x)) > 70) & (pmc["BT_X"].apply(lambda x: len(x)) < 3000), :]
    showers = pmc_to_ship_format(pmc, num_showers_in_brick=num_showers_in_brick)
    df = pd.DataFrame(showers)
    showers = gen_torch_showers(df=df, knn=knn, k=k, r=r, symmetric=symmetric, directed=directed, e=e)
    torch.save(showers, output_file)


if __name__ == "__main__":
    main()
