import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler, PowerTransformer, Normalizer


def convert(size, dataset, index):
    """
    Convert to Z representation with right indices

	Parameters:
		size (int): Size of Z
		dataset (Data): Graph datasets
		index (List): Indices to align

	Returns:
		np.array(align): Alignment representation in Z of Graph dataset
    """
    align = []
    for x in dataset:
        new_tmp = np.zeros(size)
        for i in range(len(x)):
            # Replace operation
            new_tmp[index[i]] = x[i]
        align.append(new_tmp)

    return np.array(align)


def loadData(base_f, n_dataset, fold):
    """
    Load label frequency-based graph representation

	Parameters:
		base_f (String): Path folder
		n_dataset (String): Dataset name
		fold (int): Fold number (K-Fold)

	Returns:
        x_base (np.array): r vector of the dataset
        x_train (np.array): Graph data train set
        y_train (np.array): Graph labels train set
        x_test (np.array): Graph data test set
        y_test (np.array): Graph labels test set
    """
    x_base = np.load(f"{base_f}rf_data/base_{n_dataset}_{fold}.npy")
    x_train = np.load(f"{base_f}rf_data/x_train_{n_dataset}_{fold}.npy")
    y_train = np.load(f"{base_f}rf_data/y_train_{n_dataset}_{fold}.npy")

    x_test = np.load(f"{base_f}rf_data/x_test_{n_dataset}_{fold}.npy")
    y_test = np.load(f"{base_f}rf_data/y_test_{n_dataset}_{fold}.npy")

    return x_base, x_train, y_train, x_test, y_test


def getScaler(scaler_name):
    """Get scaler"""
    if scaler_name == "RobustScaler":
        return RobustScaler()
    if scaler_name == "Normalizer":
        return Normalizer()
    if scaler_name == "PowerTransformer":
        return PowerTransformer()

    return StandardScaler()


def convertData(x_train, x_test, x_train_aligned, x_test_aligned, dim_shared, dist_aligned):
    """
    Convert datasets to Z representation

    Parameters:
        x_train (np.array): Base data train set
        x_test (np.array): Base data test set
        x_train_aligned (List(np.array)): Other data train set
        x_test_aligned (List(np.array)): Other data test set
        dim_shared (int): Size of Z
        dist_aligned (List(np.array)): Indices for all dataset (base and aligned)

    Returns:
        x_train (np.array): x_train in Z aligned representation
        x_test (np.array): x_test in Z aligned representation
        x_train_aligned (List(np.array)): x_train_aligned in Z aligned representation
        x_test_aligned (List(np.array)): x_test_aligned in Z aligned representation
    """
    x_train = convert(dim_shared, x_train, dist_aligned[0])
    x_test = convert(dim_shared, x_test, dist_aligned[0])

    for i in range(len(x_train_aligned)):
        x_train_aligned[i] = convert(dim_shared, x_train_aligned[i], dist_aligned[1][i])

    for i in range(len(x_test_aligned)):
        x_test_aligned[i] = convert(dim_shared, x_test_aligned[i], dist_aligned[1][i])

    return x_train, x_test, x_train_aligned, x_test_aligned
