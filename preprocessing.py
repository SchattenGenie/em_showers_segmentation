import numpy as np
import constants
import torch
from tqdm import tqdm


def round_Z_coodr(x):
    return constants.Z_centered[np.argmin(np.abs(constants.Z_centered - x))]


round_Z_coodr = np.vectorize(round_Z_coodr)


def create_mask(data):
    z_rounded = round_Z_coodr(data.x[:, 2].detach().cpu().numpy() * np.array([1e4]))
    orders = np.zeros((len(constants.Z_centered), data.edge_index.shape[1]))
    edge_index_0 = data.edge_index[0].detach().cpu().numpy()
    idx = np.arange(len(z_rounded))
    for i, z_i in enumerate(constants.Z_centered):
        orders[i][np.in1d(edge_index_0, idx[z_rounded == z_i])] = 1
    return orders.astype(np.uint8)


def scattering_estimation_loss(basetrack_left, basetrack_right):
    mask = basetrack_right[:, 2] < basetrack_left[:, 2]
    basetrack_left[mask], basetrack_right[mask] = basetrack_right[mask], basetrack_left[mask]

    X0 = 5 * 1000  # mm
    Es = 21  # MeV

    dz = basetrack_right[:, 2] - basetrack_left[:, 2]

    z = (
            (basetrack_right[:, 2] - basetrack_left[:, 2]) ** 2 +
            (basetrack_right[:, 1] - basetrack_left[:, 1]) ** 2 +
            (basetrack_right[:, 0] - basetrack_left[:, 0]) ** 2
    ).sqrt()
    theta_x = basetrack_right[:, 3] - basetrack_left[:, 3]
    theta_y = basetrack_right[:, 4] - basetrack_left[:, 4]
    dx = basetrack_right[:, 0] - (basetrack_left[:, 0] + basetrack_left[:, 3] * dz)
    dy = basetrack_right[:, 1] - (basetrack_left[:, 1] + basetrack_left[:, 4] * dz)

    alpha_x = 2 * theta_x ** 2 / (Es ** 2 * ((2 * z / X0).exp() - 1))
    alpha_y = 2 * theta_y ** 2 / (Es ** 2 * ((2 * z / X0).exp() - 1))

    beta_x = 24 * dx ** 2 / (Es ** 2 * X0 ** 3 * ((2 * z / X0).exp() - 1) ** 3)
    beta_y = 24 * dy ** 2 / (Es ** 2 * X0 ** 3 * ((2 * z / X0).exp() - 1) ** 3)

    gamma = 2 * (theta_x ** 2 + theta_y ** 2) / (Es ** 2 * ((2 * z / X0).exp() - 1))

    E = (3 / (alpha_x + alpha_y + beta_x + beta_y + gamma)).sqrt()

    sigma_theta = Es ** 2 * ((2 * z / X0).exp() - 1) / E ** 2
    sigma_theta_x = sigma_theta / 2
    sigma_theta_y = sigma_theta / 2

    sigma_x = Es ** 2 * ((2 * z / X0).exp() - 1) ** 3 * X0 ** 2 / (48 * E ** 2)
    sigma_y = Es ** 2 * ((2 * z / X0).exp() - 1) ** 3 * X0 ** 2 / (48 * E ** 2)

    likelihood = 0.
    likelihood -= (theta_x ** 2 / (2 * sigma_theta_x) + (sigma_theta_x).log() / 2)
    likelihood -= (theta_y ** 2 / (2 * sigma_theta_y) + (sigma_theta_y).log() / 2)

    likelihood -= (dx ** 2 / (2 * sigma_x) + (sigma_x).log() / 2)
    likelihood -= (dy ** 2 / (2 * sigma_y) + (sigma_y).log() / 2)

    likelihood -= (-(theta_x ** 2 + theta_y ** 2).log() / 2 + (sigma_theta).log() + (
                theta_x ** 2 + theta_y ** 2) / sigma_theta)
    return E, likelihood


def calculate_ips(v_data, u_data):
    v1 = (v_data[:, 0] - u_data[:, 0] - (v_data[:, 2] - u_data[:, 2]) * u_data[:, 3]) / (v_data[:, 2] - u_data[:, 2])
    v2 = (v_data[:, 1] - u_data[:, 1] - (v_data[:, 2] - u_data[:, 2]) * u_data[:, 4]) / (v_data[:, 2] - u_data[:, 2])
    v3 = (v_data[:, 0] - u_data[:, 0] - (v_data[:, 2] - u_data[:, 2]) * v_data[:, 3]) / (v_data[:, 2] - u_data[:, 2])
    v4 = (v_data[:, 1] - u_data[:, 1] - (v_data[:, 2] - u_data[:, 2]) * v_data[:, 4]) / (v_data[:, 2] - u_data[:, 2])
    return torch.cat([v1.view(-1, 1), v2.view(-1, 1), v3.view(-1, 1), v4.view(-1, 1)], dim=1)


def add_new_features(shower):
    data = shower.x
    E, likelihood = scattering_estimation_loss(data[shower.edge_index[0]], data[shower.edge_index[1]])
    ips = calculate_ips(data[shower.edge_index[0]], data[shower.edge_index[1]])
    edge_features = torch.cat([shower.edge_attr.view(-1, 1), E.view(-1, 1), likelihood.view(-1, 1), ips.view(-1, 4)],
                              dim=1)
    edge_features = edge_features / torch.tensor([1, 10, 100, 1, 1, 1, 1]).float()
    shower.edge_features = edge_features
    shape_0 = data.shape[0]
    feat_0 = torch.atan(data[:, 1] / (data[:, 0] + 0.00001)).view(shape_0, 1)
    feat_1 = (torch.sqrt(torch.pow(data[:, 1], 2) + torch.pow(data[:, 0], 2))
              / (data[:, 2] + 0.00001)).view(shape_0, 1)
    feat_2 = (data[:, 0] / (data[:, 2] + 0.00001)).view(shape_0, 1)
    feat_3 = (data[:, 1] / (data[:, 2] + 0.00001)).view(shape_0, 1)
    feat_4 = (torch.sin(feat_0) + torch.cos(feat_0)) / (feat_0 + 0.00001)
    shower.x = torch.cat([data, feat_0, feat_1, feat_2, feat_3, feat_4], dim=1)
    shower.x = shower.x / (torch.tensor([10, 10, 10, 1, 1, 1, 100, 100, 100, 1e5]).float())


def preprocess_subgraph(adj, order):
    adj_selected = adj[:, order]
    nodes_selected = adj_selected.unique()
    nodes_selected_new = torch.arange(len(nodes_selected))
    dictionary = dict(zip(nodes_selected.cpu().numpy(), nodes_selected_new.cpu().numpy()))
    adj_selected_new = torch.tensor(np.vectorize(dictionary.get)(adj_selected.cpu().numpy())).long().to(adj)
    return nodes_selected, adj_selected_new


def preprocess_dataset(datafile):
    showers = list(torch.load(datafile))
    for i in tqdm(range(len(showers))):
        orders = torch.tensor(create_mask(showers[i])).bool()
        showers[i].orders = orders
        showers[i].y = showers[i].y - showers[i].y.min() # a dirty hack to keep showers in range from 1 to MAX_SHOWERS
        orders_preprocessed = []
        for order in orders:
            if order.sum():
                nodes_selected, adj_selected_new = preprocess_subgraph(
                    adj=showers[i].edge_index,
                    order=order
                )
                orders_preprocessed.append(
                    (nodes_selected, adj_selected_new)
                )
            else:
                orders_preprocessed.append(None)
        print("orders preprocessed", len(orders_preprocessed), len(orders))
        showers[i].orders_preprocessed = orders_preprocessed
        add_new_features(showers[i])
        print(len(showers[i].x), showers[i].edge_index.shape[1])
    return showers
