import numpy as np
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from sklearn.svm import SVC
from sklearn.utils import compute_class_weight
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier as RFC
from tqdm import tqdm

from training.MultilayerP import PyTorchMLP
from training.dataManagement import convert


def optimizeModel(x_train, y_train, scaler1, model_name, dim_shared, verbose):
    """
    Optimize the model on base dataset

    Parameters:
        x_train (np.array): Graph data
        y_train (np.array): Graph labels
        scaler1 (Scaler): Scaler to use
        model_name (String): Model name
        dim_shared (int): Size of Z
        verbose: Verbose

    Returns:
        Optimize model
    """
    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))

    cv_ = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    clf = None

    # Setup up the model

    if model_name == "SVM":
        param = {"C": [0.1, 1, 10, 100], "kernel": ["poly", "rbf"], "degree": [3, 4]}
        clf = GridSearchCV(SVC(probability=True, class_weight=class_weight_dict, random_state=0), cv=cv_,
                           param_grid=param, n_jobs=-1)

    if model_name == "RFC":
        param = {"n_estimators": [100, 150, 200], "criterion": ["gini", "entropy", "log_loss"],
                 "min_samples_split": [2, 3]}
        clf = GridSearchCV(RFC(class_weight=class_weight_dict, random_state=0), cv=cv_, param_grid=param, n_jobs=-1)

    if model_name == "KNN":
        param = {"n_neighbors": [5, 7, 9], "weights": ["distance"]}
        clf = GridSearchCV(KNeighborsClassifier(), cv=cv_, param_grid=param, n_jobs=-1)

    if model_name == "MLP":
        param_grid = {'hidden_size': [32], 'lr': [0.001], 'epochs': [250]}
        clf = GridSearchCV(
            estimator=PyTorchMLP(input_size=len(x_train[0]), output_size=1, class_weight=class_weight_dict),
            param_grid=param_grid, cv=2)

    try:
        if clf is None:
            raise Exception("\n" + model_name + 'is not included in the code. Choose from '
                                                '\n-RFC: RandomForest'
                                                '\n-SVM: Support Vector Machine'
                                                '\n-KNN: K-nearest neighbors'
                                                '\n-MLP: Multilayer Perceptron '
                                                '\nor implement your own method')
    except Exception as err:
        print(err)
        return 0

    # Pre-processing
    _x_train = scaler1.fit_transform(x_train)

    # Optimize model
    if model_name != "MLP" or model_name != "SVM":
        clf.fit(_x_train, y_train)
        b_param = clf.best_params_

    if verbose:
        print(b_param)

    # Return the model
    if model_name == "SVM":
        return SVC(probability=True, C=10, kernel="rbf", degree=1, gamma="auto",
                   class_weight=class_weight_dict, random_state=0)

    if model_name == "RFC":
        return RFC(n_estimators=b_param["n_estimators"], criterion=b_param["criterion"],
                   min_samples_split=b_param["min_samples_split"], class_weight=class_weight_dict, random_state=0)

    if model_name == "KNN":
        return KNeighborsClassifier(n_neighbors=b_param["n_neighbors"], weights=b_param["weights"])

    if model_name == "MLP":
        return PyTorchMLP(input_size=dim_shared, hidden_size=32, epochs=50,
                          lr=0.01, output_size=1, class_weight=class_weight_dict)


def prepare_model(model, scaler, x, y, dim_shared, index_, base):
    """
    Train the model on base dataset

    Parameters:
        model: Machine learning model
        scaler: Scaler for pre-processing
        x (np.array): Graph data
        y (np.array):  Graph labels
        dim_shared (int): Size of Z
        index_ (np.array): indices to test
        base (np.array): if dataset is base dataset
    Returns:
        model trained on x
        scaler trained on x
    """

    x_ = []
    y_ = []

    for i in range(len(index_)):
        # Convert into Z
        x_.extend(convert(dim_shared, x[i], index_[i]))
        y_.extend(y[i])

    if base:
        x_ = scaler.fit_transform(x_)
        model.fit(x_, y_)
    else:
        x_ = scaler.transform(x_)
        model.fit(x_, y_)

    return model, scaler


def getBestIndex(model, scaler, x, y, dim_shared, index_, score):
    """
    Get the best indices list

    Parameters:
        model: Machine learning model
        scaler: Scaler for pre-processing
        x (np.array): Graph data
        y (np.array):  Graph labels
        dim_shared (int): Size of Z
        index_ (np.array): indices to test
        score (float): previous best score
    Returns:
        best_index (np.array): best indices configuration
        max_acc (float): best score
    """
    best_index = []

    max_acc = score

    # for every frequency in r
    for feature in range(len(index_)):

        # Set previous configuration
        base_t = index_
        max_index = index_[feature]

        # Restore current best configuration
        for mm in range(len(best_index)):
            base_t[mm] = best_index[mm]

        # Search best index
        for c in range(dim_shared):
            # Try a new match
            base_t[feature] = c

            # Convert data in Z
            x_ = convert(dim_shared, x, base_t)
            x_ = scaler.transform(x_)

            # Use trained model to predict
            if model.__class__ == PyTorchMLP:
                prediction = model.predict(x_)
            else:
                prediction = model.predict(x_)

            # Check accuracy
            acc_prediction = balanced_accuracy_score(y, prediction)

            # Save only if the new match improves the accuracy
            if acc_prediction > max_acc:
                max_acc = acc_prediction
                max_index = c
        best_index.append(max_index)


    return best_index, max_acc


def trainM(model, scaler1, x_train, y_train, x_train_aligned, y_train_aligned, dim_shared, K):
    """
    Get the best indices list

    Parameters:
        model: Machine learning model
        scaler1: Scaler for pre-processing
        x_train (np.array): Graph data, base dataset
        y_train (np.array): Graph labels, base dataset
        x_train_aligned (List(np.array)): Graph data, other datasets
        y_train_aligned (List(np.array)): Graph labels, other datasets
        dim_shared (int): Size of Z
        K (int): number of cycles of alignment
    Returns:
        idx_base (np.array): base indices configuration
        best_idx_datasets (np.array): best indices configuration of alignment datasets
    """
    # Set base indices
    idx_base = np.arange(len(x_train[0]))
    # Set the alignment indices
    off_set = [len(x_train[0])]
    for i in range(len(x_train_aligned) - 1):
        o_x = x_train_aligned[i]
        off_set.append(off_set[i] + len(o_x[0]))

    idx_datasets = [np.arange(len(x[0])).astype(int) for x in x_train_aligned]
    idx_datasets = [idx_datasets[i] + off_set[i] for i in range(len(idx_datasets))]
    best_idx_datasets = idx_datasets.copy()
    # Create a base accuracy values
    score_datasets = [0.5 for _ in x_train_aligned]
    old_value = 0

    count_k = 0
    # Train the model on base dataset
    model_, scaler = prepare_model(model, scaler1, [x_train.copy()], [y_train.copy()], dim_shared, [idx_base], True)
    # For K times
    while np.array(score_datasets).mean() > old_value and count_k < K:

        old_value = np.array(score_datasets).mean()
        # For every alignment datasets
        for i in tqdm(range(len(x_train_aligned))):
            x = x_train_aligned[i]
            y = y_train_aligned[i]
            # Check the best alignment
            best_index, score_datasets[i] = getBestIndex(model, scaler, x.copy(), y.copy(), dim_shared, idx_datasets[i],
                                                         score_datasets[i])

            idx_datasets[i] = best_index
        # Save the best alignment
        best_idx_datasets = idx_datasets.copy()
        print(score_datasets)
        print(f"ROC-AUC: {np.array(score_datasets).mean():.4f}")
        count_k += 1

    return idx_base, best_idx_datasets


def getDimShared(x_base, base_aligned):
    """
    Compute size of Z

    Parameters:
        x_base (np.array): r vector base dataset
        base_aligned (List(np.array)):  r vectors other datasets
    Returns:
        Size of Z (int)
    """
    dim_shared = len(x_base[0])

    for b in range(len(base_aligned)):
        dim_shared += len(base_aligned[b][0])

    print("|Z|:", dim_shared)

    return dim_shared


def getAlign(x_train, y_train, x_train_aligned, y_train_aligned, model, scaler1, K, dim_shared):
    """
        Get good indices

        Parameters:
            x_train (np.array): Graph data, base dataset
            y_train (np.array): Graph labels, base dataset
            x_train_aligned (List(np.array)): Graph data, other datasets
            y_train_aligned (List(np.array)): Graph labels, other datasets
            model: Machine learning model
            scaler1: Scaler for pre-processing
            K (int): number of cycles of alignment
            dim_shared (int): Size of Z

        Returns:
            [indices of base dataset,
            indices of other datasets]
        """
    idx_base, idx_datasets = trainM(model, scaler1, x_train, y_train, x_train_aligned,
                                    y_train_aligned, dim_shared, K)

    return [idx_base, idx_datasets]
