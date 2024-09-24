import os

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

from training.MultilayerP import PyTorchMLP


def folder_if_not_exists(path):
    """Create a folder"""
    if not os.path.exists(path):
        os.makedirs(path)


def test(model, data, label, avg_list):
    """
    Show performance.

	Parameters:
        model (Model): Machine Learning model
		data (Data): Test set data
		label (Data): Real labels test data
		avg_list (List): List of previous results
    """
    if model.__class__ == PyTorchMLP:
        prediction_roc = model.predict_proba(data)
    else:
        prediction_roc = model.predict_proba(data)
        prediction_roc = 1 - prediction_roc[:, 0]

    roc_score = roc_auc_score(label, prediction_roc)
    print(f"ROC-AUC: {roc_score:.4f}")

    prediction = model.predict(data)
    #print(label)
    #print(prediction)
    acc_prediction = balanced_accuracy_score(label, prediction)
    print(f"Accuracy: {accuracy_score(label, prediction):.4f}")
    print(f"Balanced Accuracy: {acc_prediction:.4f}")

    avg_list[0].append(acc_prediction)
    avg_list[1].append(roc_score)
    avg = np.array(avg_list[0])
    avg_random = np.array(avg_list[1])
    print(f"Avg Balanced Accuracy: {avg.mean():.4f} | Avg ROC-AUC: {avg_random.mean():.4f}")
    print(f"Std Balanced Accuracy: {avg.std():.4f} | Std ROC-AUC: {avg_random.std():.4f}")
    print("", flush=True)


def saveIndex(base, datasets, dist, path, fold, model):
    """
    Save the indices of the alignment.

	Parameters:
		base (Data): Base dataset name
		datasets (Data): Other dataset name
		dist (Numpy): Indices to save
        path (String): Path folder where to save indices
        fold (int): K-Folder number
        model (String): Model name
    """
    folder_if_not_exists(f"{path}index/")
    np.save(f"{path}index/{base}_{base}_{model}_{fold}.npy", dist[0])
    for d in range(len(dist[1])):
        np.save(f"{path}index/{base}_{datasets[d]}_{model}_{fold}.npy", dist[1][d])
