"""
Created by Constantin Philippenko, 10th January 2022.
"""

import copy
from abc import abstractmethod, ABC

import numpy as np
import torch
from sklearn.linear_model._logistic import _logistic_loss, _logistic_loss_and_grad
from sklearn.utils.extmath import log_logistic, safe_sparse_dot
from tqdm import tqdm

from scipy.special import expit

from src.CompressionModel import CompressionModel
from src.PickleHandler import pickle_saver
from src.SyntheticDataset import MAX_SIZE_DATASET
from src.Utilities import print_mem_usage

DISABLE = True


class SeriesOfSGD:

    def __init__(self, *args) -> None:
        super().__init__()
        self.series = []
        for serie in args:
            assert isinstance(serie, SGDRun), "The object added to the series is not of type SGDRun."
            self.series.append(serie)

    def append(self, *args):
        for serie in args:
            assert isinstance(serie, SGDRun), "The object added to the series is not of type SGDRun."
            self.series.append(serie)

    def save(self, filename: str):
        pickle_saver(self, filename)


class SGDRun:

    def __init__(self, last_w, losses, avg_losses, diag_cov_gradients, label=None) -> None:
        super().__init__()
        self.last_w = last_w
        self.losses = losses
        self.avg_losses = avg_losses
        self.diag_cov_gradients = diag_cov_gradients
        self.label = label


class SGD(ABC):
    NB_EPOCH = 1
    
    def __init__(self, synthetic_dataset) -> None:
        super().__init__()
        self.do_logistic_regression = synthetic_dataset.do_logistic_regression
        self.synthetic_dataset = synthetic_dataset
        self.X, self.Y = self.synthetic_dataset.X, self.synthetic_dataset.Y
        self.w_star = self.synthetic_dataset.w_star
        self.GAMMA = self.synthetic_dataset.gamma
        self.SIZE_DATASET, self.DIM = self.synthetic_dataset.size_dataset, self.synthetic_dataset.dim
        np.random.seed(25)
        self.w0 = np.random.normal(0, 1, size = self.DIM)
        self.additive_stochastic_gradient = False

    def compute_empirical_risk(self, w, data, labels):
        if self.do_logistic_regression:
            return -np.sum(np.log(expit(labels * (data @ w)))) / len(labels)
        return 0.5 * np.linalg.norm(data @ w - labels) ** 2 / len(labels)

    def compute_true_risk(self, w, data, labels):
        if self.do_logistic_regression:
            return -np.sum(log_logistic(labels * (data @ w))) / len(labels)
        return 0.5 * (w - self.w_star).T @ self.synthetic_dataset.upper_sigma @ (w - self.w_star)

    def compute_stochastic_gradient(self, w, data, labels, index):
        x, y = data[index], labels[index]
        if self.do_logistic_regression:
            s = expit(y * x @ w)
            return x * ((s - 1) * y)
        return np.array((x @ w - y)).dot(x)

    def compute_additive_stochastic_gradient(self, w, data, labels, index):
        x, y = data[index], labels[index]
        if self.additive_stochastic_gradient:
            raise ValueError("Compute only additive stochastic gradient is not possible in the logistic setting")
        return self.synthetic_dataset.upper_sigma.dot(w) - y * x

    def sgd_update(self, w, gradient, gamma):
        return w - gamma * gradient

    def gradient_descent(self, label: str = None) -> SGDRun:
        current_w = self.w0
        avg_w = copy.deepcopy(current_w)
        it = 1
        losses = [self.compute_empirical_risk(current_w, self.X, self.Y)]
        avg_losses = [self.compute_empirical_risk(avg_w, self.X, self.Y)]
        matrix_grad = np.zeros((self.SIZE_DATASET, self.DIM))
        for epoch in range(self.NB_EPOCH):
            indices = np.arange(self.SIZE_DATASET)
            for idx in tqdm(indices, disable=DISABLE):
                if idx % MAX_SIZE_DATASET == 0 and idx != 0:
                    print("Regenerating ...")
                    self.synthetic_dataset.regenerate_dataset()
                gamma = self.synthetic_dataset.gamma
                it += 1
                if self.additive_stochastic_gradient:
                    grad = self.compute_additive_stochastic_gradient(current_w, self.X, self.Y, idx)
                else:
                    grad = self.compute_stochastic_gradient(current_w, self.X, self.Y, idx)
                g = self.gradient_processing(grad)
                matrix_grad[idx] = g
                current_w = self.sgd_update(current_w, g, gamma)
                avg_w = current_w / it + avg_w * (it - 1) / it
                losses.append(self.compute_true_risk(current_w, self.X, self.Y))
                avg_losses.append(self.compute_true_risk(avg_w, self.X, self.Y))

        matrix_cov = matrix_grad.T.dot(matrix_grad) / self.SIZE_DATASET
        print_mem_usage("End of sgd descent ...")
        return SGDRun(current_w, losses, avg_losses, np.diag(matrix_cov), label=label)

    @abstractmethod
    def gradient_processing(self, grad):
        pass


class SGDVanilla(SGD):

    def gradient_processing(self, grad):
        return grad


class SGDNoised(SGD):

    def gradient_processing(self, grad):
        return grad + np.random.normal(0, 1, size=self.DIM)


class SGDCompressed(SGD):

    def __init__(self, synthetic_dataset, compressor: CompressionModel) -> None:
        super().__init__(synthetic_dataset)
        self.compressor = compressor

    def gradient_processing(self, grad):
        return self.compressor.compress(grad)



