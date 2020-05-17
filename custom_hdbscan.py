from functools import total_ordering
import operator
from collections import Counter, defaultdict
from tqdm import tqdm
import networkx as nx
import numpy as np
import hdbscan


@total_ordering
class ClusterHDBSCAN(object):
    def __init__(self, weight: float, cl_size: int, clusters: list = None, nodes: list = None):
        # init
        self.nodes = set()
        self.nodes_in = Counter()
        self.nodes_out = Counter()

        self.weights_nodes_dict = defaultdict(set)

        self.weight_death = weight
        self.lambda_death = 1. / (weight + 1e-5)

        self.weight_birth = weight
        self.lambda_birth = 1. / (weight + 1e-5)

        self.children = []
        self.falling_out_points = []

        assert clusters is not None or nodes is not None
        if clusters is not None:
            for cluster in clusters:
                self.nodes.update(cluster.nodes)
                self.nodes_in.update(cluster.nodes_in)
                self.nodes_out.update(cluster.nodes_out)
                self.weights_nodes_dict[weight].update(cluster.nodes)
                if cluster.is_cluster:
                    cluster.set_weight_birth(weight)
                    self.children.append(cluster)
                else:
                    self.falling_out_points.append(cluster)
        else:
            self.nodes.update(nodes)
            self.nodes_out.update(nodes)
            self.weights_nodes_dict[weight].update(nodes)
        self.frozennodes = frozenset(self.nodes)
        self.__hash = hash(self.frozennodes)
        self.listnodes = list(self.nodes)
        self.npnodes = np.array(list(self.nodes)).astype(np.int32)
        self.cl_size = cl_size
        self.is_cluster = len(self) >= cl_size
        self.is_noise = not self.is_cluster
        self.stability = None

    def append(self, weight: float, clusters: list):
        """
        Adding
        """
        for cluster in clusters:
            self.nodes.update(cluster.nodes)
            self.weights_nodes_dict[weight].update(cluster.nodes)
        self.weight_birth = weight
        self.lambda_birth = 1 / (weight + 1e-5)
        self.frozennodes = frozenset(self.nodes)
        self.__hash = hash(self.frozennodes)
        self.listnodes = list(self.nodes)
        self.npnodes = np.array(list(self.nodes)).astype(np.int32)
        self.is_cluster = len(self) >= self.cl_size
        self.is_noise = not self.is_cluster
        return self

    def __iter__(self):
        for child in self.children:
            yield child

    def __contains__(self, node):
        return node in self.nodes

    def __len__(self):
        return len(self.nodes)

    def __hash__(self):
        return self.__hash

    def __eq__(self, other):
        return self.__hash == other.__hash

    def __lt__(self, other):
        return self.__hash < other.__hash

    def set_weight_birth(self, weight: float):
        self.weight_birth = weight
        self.lambda_birth = 1. / (weight + 1e-5)

    def calculate_stability(self):
        self.stability = 0.
        self.lambda_birth = 1. / (max(self.weights_nodes_dict.keys()) + 1e-5)
        # norm = self.lambda_birth len(self.weights_nodes_dict[weight]) *
        for weight in self.weights_nodes_dict:
            self.stability += (1 / (weight + 1e-5) - self.lambda_birth)  # * len(self.weights_nodes_dict[weight]) * norm


def calc_stabilities(root):
    root.calculate_stability()
    for child in root:
        calc_stabilities(child)


def class_disbalance(cluster, graph):
    subgraph = graph.subgraph(cluster.nodes)
    signal = []
    for _, node in subgraph.nodes(data=True):
        signal.append(node['signal'])
    return list(zip(*np.unique(signal, return_counts=True)))


def flat_clusters(root):
    if root.is_cluster:
        yield root

    for child in root:
        for cluster in flat_clusters(child):
            yield cluster


def reed_stabilities(root, level=0):
    print('    ' * (level - 1) + '+---' * (level > 0), end='')
    print('len={}'.format(len(root)), end=' ')
    print('stability={:.2f}'.format(root.stability))
    for child in root:
        reed_stabilities(child, level + 1)


def print_class_disbalance_for_all_clusters(root, graph, level=0):
    class_disbalance_tuples = class_disbalance(root, graph)

    print('    ' * (level - 1) + '+---' * (level > 0), end='')
    print('len={}'.format(len(root)))
    print('    ' * (level), end='')
    print(class_disbalance_tuples, end=' ')
    print('stability={:.3f}'.format(root.stability))
    for child in root:
        print_class_disbalance_for_all_clusters(child, graph, level + 1)


def leaf_clusters(root):
    if root.is_cluster and len(root.children) == 0:
        yield root

    for child in root:
        for cluster in leaf_clusters(child):
            yield cluster


def max_level_clusters(root, level=0, max_level=2):
    if level == max_level and root.is_cluster:
        yield root

    for child in root:
        for cluster in max_level_clusters(child, level=level + 1, max_level=max_level):
            yield cluster


def recalc_tree(root):
    weights_children = 0.
    for child in root:
        weights_children += recalc_tree(child)
    if weights_children > root.stability:
        root.stability = weights_children
    else:
        root.children.clear()

    return root.stability


def mutual_reachability(distance_matrix, min_points=5, alpha=1.0):
    size = distance_matrix.shape[0]
    min_points = min(size - 1, min_points)
    try:
        core_distances = np.partition(distance_matrix,
                                      min_points,
                                      axis=0)[min_points]
    except AttributeError:
        core_distances = np.sort(distance_matrix,
                                 axis=0)[min_points]

    if alpha != 1.0:
        distance_matrix = distance_matrix / alpha

    stage1 = np.where(core_distances > distance_matrix,
                      core_distances, distance_matrix)
    result = np.where(core_distances > stage1.T,
                      core_distances.T, stage1.T).T
    return result


def run_hdbscan(G, cl_size=20, min_samples_core=5, order=True):
    # core d distances
    G_old = nx.Graph(G)
    adj = nx.adjacency_matrix(G_old).toarray()
    adj[adj == 0] = np.inf
    adj = mutual_reachability(adj, min_points=min_samples_core)
    nodes_adj_map = {n: i for i, n in enumerate(G.nodes())}
    for i, j, edge in G.edges(data=True):
        edge["weight"] = adj[nodes_adj_map[i], nodes_adj_map[j]]

    # % % time
    G = nx.minimum_spanning_tree(nx.Graph(G))
    # core_d was deleted => could be returned. Leverage robustness / cluster sharpness.
    edges = []
    for node_id_left, node_id_right, edge in G.edges(data=True):
        node_left = G.nodes[node_id_left]
        node_right = G.nodes[node_id_right]
        edges.append(
            (
                node_id_left,
                node_id_right,
                edge['weight'],
                np.sign(node_left['features']['SZ'] - node_right['features']['SZ'])
            )
        )

    edges = sorted(edges, key=operator.itemgetter(2))

    # Minimum spanning tree was also thrown
    # following algo reminds of Kruskal algo but with some modifications

    # init
    clusters = {}
    linkage_tree = []

    current_id = 0
    for node_id in G.nodes():
        clusters[node_id] = (ClusterHDBSCAN(cl_size=cl_size, weight=np.inf, nodes=[node_id]), current_id)
        current_id += 1
    current_id -= 1

    for i, j, weight, *_ in tqdm(edges):
        cluster_out, cluster_out_id = clusters[i]
        cluster_in, cluster_in_id = clusters[j]

        if cluster_in is cluster_out:
            continue
        current_id += 1

        if cluster_in.is_cluster and cluster_out.is_cluster:
            cluster = ClusterHDBSCAN(weight=weight, cl_size=cl_size, clusters=[cluster_in, cluster_out])
        elif cluster_in.is_cluster and not cluster_out.is_cluster:
            cluster = cluster_in.append(weight=weight, clusters=[cluster_out])
        elif cluster_out.is_cluster and not cluster_in.is_cluster:
            cluster = cluster_out.append(weight=weight, clusters=[cluster_in])
        else:
            cluster = ClusterHDBSCAN(weight=weight, cl_size=cl_size, clusters=[cluster_in, cluster_out])

        cluster.nodes_out[i] += 1
        cluster.nodes_in[j] += 1

        clusters.update({l: (cluster, current_id) for l in cluster.nodes})
        linkage_tree.append([cluster_in_id, cluster_out_id, weight, len(cluster)])

    clusters = list(set(clusters.values()))

    # choose biggest cluster
    root = clusters[0][0]
    length = len(clusters[0][0])
    for cluster in clusters:
        cluster = cluster[0]
        if len(cluster) > length:
            length = len(cluster)
            root = cluster

    calc_stabilities(root)
    recalc_tree(root)
    clusters = list(leaf_clusters(root))
    return clusters, root, linkage_tree


def run_hdbscan_on_brick(graphx, min_cl=40, cl_size=40, min_samples_core=5, order=True):
    connected_components = []
    for cnn in nx.connected_components(nx.Graph(graphx)):
        if len(cnn) > min_cl:
            connected_components.append(nx.DiGraph(graphx.subgraph(cnn)))
    clusters = []
    roots = []
    for G in tqdm(connected_components):
        if len(G) < 100:
            clusters.append(G)
        else:
            clusters_hdbscan, root_hdbscan, _ = run_hdbscan(G, cl_size=cl_size, order=order, min_samples_core=min_samples_core)
            roots.append(root_hdbscan)
            clusters.extend(clusters_hdbscan)

    return graphx, clusters, roots

def run_vanilla_hdbscan(G, cl_size, min_samples_core=5, order=None):
    distance_matrix = nx.adjacency_matrix(nx.Graph(G)).toarray().astype(np.float64)
    distance_matrix[distance_matrix == 0.] = np.inf
    clusterer = hdbscan.HDBSCAN(min_cluster_size=cl_size,
                                min_samples=min_samples_core,
                                metric="precomputed",
                                core_dist_n_jobs=-1)  # precomputed
    clusterer.fit(distance_matrix)
    clusters = []
    for lab in np.unique(clusterer.labels_):
        clusters.append(G.subgraph(np.array(list(G.nodes))[clusterer.labels_ == lab]))
    return clusters, None


def run_vanilla_hdbscan_on_brick(graphx, min_cl=40, cl_size=40, min_samples_core=5, order=True):
    connected_components = []
    for cnn in nx.connected_components(nx.Graph(graphx)):
        if len(cnn) > min_cl:
            connected_components.append(nx.DiGraph(graphx.subgraph(cnn)))
    clusters = []
    roots = []
    for G in tqdm(connected_components):
        if len(G) < 100:
            clusters.append(G)
        else:
            clusters_hdbscan, root_hdbscan = run_vanilla_hdbscan(G, cl_size=cl_size, order=order,
                                                                 min_samples_core=min_samples_core)
            roots.append(root_hdbscan)
            clusters.extend(clusters_hdbscan)

    return graphx, clusters, roots
