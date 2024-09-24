"""
Development of a custom MultiLayer Perceptron
"""

import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set seed
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.LeakyReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc3(out)
        out = torch.nn.Sigmoid()(out)
        return out


class PyTorchMLP(BaseEstimator, ClassifierMixin):
    def __init__(self, input_size=53, hidden_size=64, output_size=1, lr=0.001, epochs=500, class_weight=None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr
        self.class_weight = class_weight
        self.epochs = epochs
        self.model = None

    def build_model(self):
        self.model = MLP(self.input_size, self.hidden_size, self.output_size).to(device)
        self.criterion = nn.BCELoss(reduction='none')
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def fit(self, X, y, sample_weight=None, batch_=64):
        self.build_model()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

        if sample_weight is not None:
            sample_weight_tensor = torch.tensor(sample_weight, dtype=torch.float32).view(-1, 1)
        else:
            sample_weight_tensor = torch.ones_like(y_tensor, dtype=torch.float32)

        if self.class_weight is not None:
            class_weights = torch.tensor([self.class_weight[int(cls)] for cls in y], dtype=torch.float32).view(-1, 1)
            sample_weight_tensor *= class_weights

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor, sample_weight_tensor)
        generator = torch.Generator().manual_seed(0)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_, shuffle=True, generator=generator)

        for _ in tqdm(range(self.epochs)):
            for data_x, labels, weights in dataloader:
                data_x = data_x.view(-1, self.input_size).to(device)
                outputs = self.model(data_x)
                loss = self.criterion(outputs, labels.to(device))
                loss = (loss * weights.to(device)).mean()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return self

    def predict(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            predictions = self.model(X_tensor.view(-1, self.input_size).to(device))
            predicted_classes = (predictions >= 0.5).float()
        return predicted_classes.cpu().numpy().flatten()

    def predict_proba(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            predictions = self.model(X_tensor.view(-1, self.input_size).to(device))
        return predictions.cpu().numpy().flatten()

    def score(self, X, y):
        y_pred = self.predict(X)

        a = np.array(y)
        b = np.array(y_pred)

        return balanced_accuracy_score(a, b)
