import torch
import random
import argparse
import numpy as np

from tqdm import tqdm
from matplotlib import pyplot as plt
from utils import folder_if_not_exists
from sklearn.model_selection import StratifiedKFold

from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset


def set_seed(seed):
    """Set a seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(0)

base_path = "./"


def details(dataset):
    """Show details of a dataset"""
    nodes = 0
    edges = 0
    classes = dataset[0].x.shape[-1]

    d_flat = []
    degree_flag = False

    if classes == 1:
        degree_flag = True
        d_flat = []

    for d in dataset:
        if degree_flag:
            d_flat.extend([xi.item() for xi in d.x])

        nodes += d.x.shape[0]
        edges += d.edge_index.shape[1]
    print()
    print("Nodes: ", round(nodes / len(dataset), 2))
    print("Edges: ", round(edges / len(dataset) / 2, 2))
    print("Graphs:", len(dataset))
    if degree_flag:
        print("Classes per Node: ", len(set(d_flat)))
    else:
        print("Classes per Node: ", classes)
    print()


def downloadDataset(name, degree_):
    """
     Download the dataset.

             Parameters:
                     name (String): Dataset name
                     degree_ (bool): Use of degree as label

             Returns:
                     dataset: Dataset
                     y_dataset: Graph labels
    """
    try:
        dataset = TUDataset(root=base_path + 'TUDataset', name=name, use_node_attr=False)

        dataset = [d for d in dataset]
        random.shuffle(dataset)

        if degree_:
            new_dataset = []
            for data in dataset:
                nodes, count = torch.unique(data.edge_index[0], return_counts=True)
                new_dataset.append(Data(x=count.unsqueeze(-1), edge_index=data.edge_index, y=data.y))

            dataset = new_dataset

        y_dataset = [d.y.item() for d in dataset]

        return dataset, y_dataset
    except:
        print("Problem downloading from TUDataset")


def getData(dataset, y_dataset, n_dataset, r_threshold, reverse, verbose):
    """
     Convert the dataset.

             Parameters:
                     dataset (Data): Graph data
                     y_dataset (array): Graph labels
                     n_dataset (String): Dataset name
                     r_threshold (int): Round factor
                     reverse (bool): Reverse y_dataset
                     verbose (bool): Verbose

             Returns:
                     dataset: Dataset
                     y_dataset: Graph labels
    """
    for fold, (train_index, test_index) in enumerate(
            StratifiedKFold(n_splits=10, shuffle=True, random_state=0).split(dataset, y_dataset)):

        print("Fold:", fold)

        train_set = [dataset[i] for i in train_index]
        test_set = [dataset[i] for i in test_index]

        if verbose:
            details(dataset)

        if reverse:
            new_train = []
            new_test = []
            for i in range(len(train_set)):
                tmp = train_set[i].clone()
                tmp.y = abs(1 - train_set[i].y)
                new_train.append(tmp)
            for i in range(len(test_set)):
                tmp = test_set[i].clone()
                tmp.y = abs(1 - test_set[i].y)
                new_test.append(tmp)

            train_set = new_train
            test_set = new_test

        # Split data based on the classes of datasets
        g_pos = [t for t in train_set if t.y == 1]
        g_neg = [t for t in train_set if t.y == 0]

        # Get labels of nodes
        x_pos = np.array([x.numpy() for t in g_pos for x in t.x])
        x_neg = np.array([x.numpy() for t in g_neg for x in t.x])

        # Count the elements
        u_pos, c_pos = np.unique(x_pos, return_counts=True, axis=0)
        u_neg, c_neg = np.unique(x_neg, return_counts=True, axis=0)

        # Total number of graphs
        tot = np.sum(c_pos) + np.sum(c_neg)
        # Diz -> (label: [class positive, class negative])
        diz_labels_classes = {}

        for i in range(len(u_pos)):
            if len(u_pos[i]) > 1:
                idx = np.argmax(u_pos[i])
            else:
                idx = u_pos[i][0]

            diz_labels_classes[idx] = [c_pos[i]]

        for i in range(len(u_neg)):
            if len(u_neg[i]) > 1:
                idx = np.argmax(u_neg[i])
            else:
                idx = u_neg[i][0]

            if idx in diz_labels_classes:
                diz_labels_classes[idx].append(c_neg[i])
            else:
                diz_labels_classes[idx] = [0, c_neg[i]]

        nodo = []
        relative_fs = []
        imbalance = []

        diz_relative_freq = {}
        for d in diz_labels_classes:
            nodo.append(d)
            val = diz_labels_classes[d]

            nodo_rar = round(sum(val) / tot, r_threshold)

            diz_relative_freq[d] = nodo_rar

            relative_fs.append(nodo_rar)
            imbalance.append(val)

        diz_finale = {}
        for i in range(len(relative_fs)):
            r = relative_fs[i]
            p = imbalance[i]
            if r in diz_finale:
                val_d = diz_finale[r]
                val_d[0] += p[0]
                if len(p) > 1:
                    val_d[1] += p[1]
                diz_finale[r] = val_d
            else:
                diz_finale[r] = [0, 0]
                val_d = diz_finale[r]
                val_d[0] += p[0]
                if len(p) > 1:
                    val_d[1] += p[1]
                diz_finale[r] = val_d

        relative_fs_final = []
        imbalance_final = []

        for d in diz_finale:
            relative_fs_final.append(d)
            val_d = diz_finale[d]
            val_d = round((val_d[0] - val_d[1]) / sum(val_d), 4)
            imbalance_final.append(val_d)

        relative_fs_final = np.array(relative_fs_final)
        imbalance_final = np.array(imbalance_final)

        sort_idx = np.argsort(relative_fs_final)
        relative_fs_final = relative_fs_final[sort_idx]
        imbalance_final = imbalance_final[sort_idx]

        # Vector R
        diz_sort_R = {}
        count_pos = 0
        for r in relative_fs_final:
            diz_sort_R[r] = count_pos
            count_pos += 1

        if verbose:
            print("Relative Frequencies")
            print(relative_fs_final)
            print("Normalized Imbalance")
            print(imbalance_final)

            print("r:", diz_sort_R)

        folder_if_not_exists(f"{base_path}rf_plot")

        ax = plt.gca()
        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-0.1, 1.1])
        ax.vlines(x=0, ymin=-1, ymax=1.5)

        plt.Figure(figsize=(20, 20))
        plt.title(n_dataset)
        plt.scatter(imbalance_final, relative_fs_final, s=50, edgecolors='black')

        for txt in range(len(imbalance_final)):
            plt.annotate(txt, (imbalance_final[txt], relative_fs_final[txt]))

        if reverse:
            plt.savefig(f"{base_path}rf_plot/{n_dataset}_{fold}_reverse.png")
        else:
            plt.savefig(f"{base_path}rf_plot/{n_dataset}_{fold}.png")
        plt.close()

        train_x = []
        train_y = []

        for t in tqdm(train_set):

            x_converted = np.zeros(len(imbalance_final))
            for x in t.x:

                if len(x) > 1:
                    idx = np.argmax(x).item()
                else:
                    idx = x.item()

                try:
                    rar = diz_relative_freq[idx]
                    pos = diz_sort_R[rar]
                    x_converted[pos] += 1
                except:
                    x_converted[0] += 1

            train_x.append(x_converted)
            train_y.append(t.y.item())

        test_x = []
        test_y = []
        for t in tqdm(test_set):

            x_converted = np.zeros(len(imbalance_final))
            for x in t.x:

                if len(x) > 1:
                    idx = np.argmax(x).item()
                else:
                    idx = x.item()

                try:
                    rar = diz_relative_freq[idx]
                    pos = diz_sort_R[rar]
                    x_converted[pos] += 1
                except:
                    x_converted[0] += 1

            test_x.append(x_converted)
            test_y.append(t.y.item())

        folder_if_not_exists(f"{base_path}rf_data")

        base_align = np.concatenate((relative_fs_final, imbalance_final), 0).reshape(2, -1)

        np.save(f"{base_path}rf_data/base_{n_dataset}_{fold}.npy", base_align)
        np.save(f"{base_path}rf_data/x_train_{n_dataset}_{fold}.npy", train_x)
        np.save(f"{base_path}rf_data/y_train_{n_dataset}_{fold}.npy", train_y)

        np.save(f"{base_path}rf_data/x_test_{n_dataset}_{fold}.npy", test_x)
        np.save(f"{base_path}rf_data/y_test_{n_dataset}_{fold}.npy", test_y)
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parameters to pass for creating datasets as relative frequencies with StratifiedKFold")

    parser.add_argument('-d', '--dataset', type=str, required=True, default="MUTAG",
                        help="Dataset name TUDortmund (https://github.com/nd7141/graph_datasets/tree/master/datasets)")

    parser.add_argument('-round', '--round', type=int, default=4,
                        help="Decimal digit to round (Default: 4)")

    parser.add_argument('-degree', '--degree', action='store_true',
                        help="Whether to turn labels into degrees (required for social networks)")

    parser.add_argument('-r', '--reverse', action='store_true',
                        help="Reverse the classes of the graphs")

    parser.add_argument('-v', '--verbose', action='store_true', help="Enable verbose output")

    args = parser.parse_args()

    print("Dataset: ", args.dataset)
    print("Degree:  ", args.degree)
    print("Rounding:", args.round)
    print("Reverse: ", args.reverse)
    print()

    dataset_, y_dataset_ = downloadDataset(args.dataset, args.degree)

    getData(dataset_, y_dataset_, args.dataset, args.round, args.reverse, args.verbose)
