"""
Created by Constantin Philippenko, 10th January 2022.
"""

import copy
import math
from abc import abstractmethod, ABC

from matplotlib import pyplot as plt
from numba import jit
import numpy as np
from numpy.linalg import inv
from scipy.linalg import sqrtm
from tqdm import tqdm

from scipy.special import expit

from src.CompressionModel import CompressionModel
from src.JITProduct import *
from src.PickleHandler import pickle_saver
from src.SyntheticDataset import MAX_SIZE_DATASET
from src.Utilities import print_mem_usage

ONLY_ADDITIVE_NOISE = False
USE_MOMENTUM = False
BETA = 0
REGULARIZATION = 0

DISABLE = False
CORRECTION_SQUARE_COV = False
CORRECTION_DIAG = False


def log_sampling_xaxix(size_dataset):
    log_len = np.int(math.log10(size_dataset))
    residual_len = math.log10(size_dataset) - log_len
    log_xaxis = [[math.pow(10, a) * math.pow(10, i / 100) for i in range(100)] for a in range(log_len)]
    log_xaxis.append([math.pow(10, log_len) * math.pow(10, i / 100) for i in range(int(100 * residual_len))])
    log_xaxis = np.concatenate(log_xaxis, axis=None)
    log_xaxis = np.unique(log_xaxis.astype(int))
    return log_xaxis


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

    def __init__(self, size_dataset, last_w, losses, avg_losses, diag_cov_gradients, label=None) -> None:
        super().__init__()
        self.size_dataset = size_dataset
        self.last_w = last_w
        self.losses = np.array(losses)
        self.avg_losses = np.array(avg_losses)
        self.log_xaxis = log_sampling_xaxix(size_dataset)
        self.diag_cov_gradients = diag_cov_gradients
        self.label = label


class SGD(ABC):
    NB_EPOCH = 1
    
    def __init__(self, synthetic_dataset, reg: int = REGULARIZATION) -> None:
        super().__init__()
        np.random.seed(25)
        self.do_logistic_regression = synthetic_dataset.do_logistic_regression
        self.synthetic_dataset = synthetic_dataset
        self.w_star = self.synthetic_dataset.w_star
        self.SIZE_DATASET, self.DIM = self.synthetic_dataset.size_dataset, self.synthetic_dataset.dim
        self.additive_stochastic_gradient = ONLY_ADDITIVE_NOISE
        self.root_square_upper_sigma = sqrtm(self.synthetic_dataset.upper_sigma)
        self.inv_root_square_upper_sigma = inv(self.root_square_upper_sigma)

        self.Q, self.D = np.identity(self.DIM), np.identity(self.DIM)
        self.approx_hessian = np.identity(self.DIM)
        self.debiased_hessian = np.identity(self.DIM)
        self.reg = reg
        self.compressor = None

    def compute_empirical_risk(self, w, data, labels):
        # if self.do_logistic_regression:
        #     return -np.sum(np.log(expit(labels * (data @ w)))) / len(labels)
        if CORRECTION_SQUARE_COV:
            data = data @ self.inv_root_square_upper_sigma.T
        if CORRECTION_DIAG:
            data = data @ self.Q
        return 0.5 * np.linalg.norm(data @ w - labels) ** 2 / len(labels)

    def compute_true_risk(self, w, data, labels):
        if data is None:
            return 0
        # data = data @ self.transition_matrix.T
        # if self.do_logistic_regression:
        #     return -np.sum(log_logistic(labels * (data @ w))) / len(labels)
        # w = self.inv_transition_matrix @ w
        if CORRECTION_SQUARE_COV:
            w_star = self.inv_root_square_upper_sigma @ self.w_star
            return constant_product(0.5, vectorial_norm(minus(w, w_star)))
        elif CORRECTION_DIAG:
            w_star = matrix_vector_product(self.Q.T, self.w_star)
            return wAw_product(0.5, minus(w, w_star), self.D)
        return wAw_product(0.5, minus(w, self.w_star), self.synthetic_dataset.upper_sigma)

    def compute_stochastic_gradient(self, w, data, labels, index):
        x, y = data[index], labels[index]
        if CORRECTION_SQUARE_COV:
            x = matrix_vector_product(self.inv_root_square_upper_sigma, x)
        elif CORRECTION_DIAG:
            x = matrix_vector_product(self.Q.T, x)
        return (x @ w - y) * x

    def compute_additive_stochastic_gradient(self, w, data, labels, index):
        x, y = data[index], labels[index]
        if self.additive_stochastic_gradient and self.do_logistic_regression:
            raise ValueError("Compute only additive stochastic gradient is not possible in the logistic setting")
        if CORRECTION_SQUARE_COV:
            x = matrix_vector_product(self.inv_root_square_upper_sigma, x)
        elif CORRECTION_DIAG:
            x = matrix_vector_product(self.Q.T, x)
        return self.D.dot(w) - y * x

    def sgd_update(self, w, gradient, gamma):
        return w - gamma * gradient - self.reg * (w - self.synthetic_dataset.w0)

    def gradient_descent(self, label: str = None) -> SGDRun:
        log_xaxis = log_sampling_xaxix(self.synthetic_dataset.size_dataset)
        current_w = self.synthetic_dataset.w0
        avg_w = copy.deepcopy(current_w)
        it = 1
        losses = [self.compute_true_risk(current_w, self.synthetic_dataset.X_complete, self.synthetic_dataset.Y)]
        avg_losses = [losses[-1]]
        for epoch in range(self.NB_EPOCH):
            indices = np.arange(self.SIZE_DATASET)
            for idx in tqdm(indices, disable=DISABLE):
                if idx % MAX_SIZE_DATASET == 0 and idx != 0:
                    print("Regenerating ...")
                    self.synthetic_dataset.regenerate_dataset()
                gamma = self.synthetic_dataset.gamma
                it += 1

                if CORRECTION_DIAG:
                    self.Q, self.D = self.synthetic_dataset.Q, self.synthetic_dataset.D #diagonalization(self.debiased_hessian) #

                if self.additive_stochastic_gradient:
                    grad = self.compute_additive_stochastic_gradient(current_w, self.synthetic_dataset.X_complete,
                                                                     self.synthetic_dataset.Y, idx % MAX_SIZE_DATASET)
                else:
                    grad = self.compute_stochastic_gradient(current_w, self.synthetic_dataset.X_complete,
                                                            self.synthetic_dataset.Y, idx % MAX_SIZE_DATASET)
                grad = self.gradient_processing(grad)

                if idx == 0:
                    self.approx_hessian = np.kron(grad, grad).reshape((self.DIM, self.DIM))
                else:
                    self.approx_hessian = np.kron(grad, grad).reshape((self.DIM, self.DIM)) / it + self.approx_hessian * (it - 1)/ it

                current_w = self.sgd_update(current_w, grad, gamma)
                avg_w = current_w / it + avg_w * (it - 1) / it
                if idx in log_xaxis[1:]:
                    losses.append(self.compute_true_risk(current_w, self.synthetic_dataset.X_complete, self.synthetic_dataset.Y))
                    avg_losses.append(self.compute_true_risk(avg_w, self.synthetic_dataset.X_complete, self.synthetic_dataset.Y))

        print_mem_usage("End of sgd descent ...")

        if self.synthetic_dataset.use_ortho_matrix:
            cov_matrix = self.synthetic_dataset.ortho_matrix.T.dot(self.approx_hessian).dot(self.synthetic_dataset.ortho_matrix)
        else:
            cov_matrix = self.approx_hessian

        return SGDRun(self.synthetic_dataset.size_dataset, current_w, losses, avg_losses, np.diag(cov_matrix), 
                      label=label)
        # return SGDRun(current_w, losses, avg_losses, np.diag(self.synthetic_dataset.Q.T @ self.approx_hessian @ self.synthetic_dataset.Q), label=label)

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
        # if isinstance(compressor, RandomSparsification):
        #     p = compressor.level
        #     self.inv_proba_matrix = np.eye(self.DIM) - (1 - p) * np.identity(self.DIM)

    def gradient_processing(self, grad):
        return self.compressor.decompress(self.compressor.compress(grad))


class SGDSportisse(SGD):

    def __init__(self, synthetic_dataset, compressor: CompressionModel) -> None:
        super().__init__(synthetic_dataset)
        self.compressor = compressor

    def gradient_processing(self, grad):
        return grad

    def compute_stochastic_gradient(self, w, data, labels, index):
        """Can be used only in the MISSING VALUE MODE."""
        x, y = self.synthetic_dataset.X[index], self.synthetic_dataset.Y[index]
        p = self.synthetic_dataset.estimated_p
        g = x * (w @ x - y) - (1 - p) * np.diag(x ** 2) @ w
        return g


class SGDNaiveSparsification(SGDCompressed):

    def gradient_processing(self, grad):
        return self.compressor.compress(grad)

    def gradient_processing(self, grad):
        return grad

    def compute_stochastic_gradient(self, w, data, labels, index):
        x, y = data[index], labels[index]
        x = self.synthetic_dataset.sparsificator.compress(x)
        g = x * (w @ x - y)
        return g

