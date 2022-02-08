"""
Created by Constantin Philippenko, 10th January 2022.
"""

import copy
from abc import abstractmethod, ABC

from matplotlib import pyplot as plt
from numba import jit
import numpy as np
from numpy.linalg import inv
from scipy.linalg import sqrtm
from tqdm import tqdm

from scipy.special import expit

from src.CompressionModel import CompressionModel, RandomSparsification
from src.JITProduct import *
from src.PickleHandler import pickle_saver
from src.SyntheticDataset import MAX_SIZE_DATASET
from src.Utilities import print_mem_usage

DISABLE = False
CORRECTION_SQUARE_COV = False
CORRECTION_DIAG = False

# CORRECTION_ORTHO = None
# "The correction in the orthogonal case must be in: None, 'square_cov', 'diagonalization'."


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
        self.root_square_upper_sigma = sqrtm(self.synthetic_dataset.upper_sigma)
        self.inv_root_square_upper_sigma = inv(self.root_square_upper_sigma)

        self.Q, self.D = np.identity(self.DIM), np.identity(self.DIM)
        self.approx_hessian = np.identity(self.DIM)
        self.debiased_hessian = np.identity(self.DIM)

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
        # if self.do_logistic_regression:
        #     s = expit(y * x @ w)
        #     return x * ((s - 1) * y)
        if CORRECTION_SQUARE_COV:
            x = matrix_vector_product(self.inv_root_square_upper_sigma, x)
        elif CORRECTION_DIAG:
            x = matrix_vector_product(self.Q.T, x)
        return constant_product(minus(scalar_product(x, w), y), x)

    def compute_additive_stochastic_gradient(self, w, data, labels, index):
        x, y = data[index], labels[index]
        if self.additive_stochastic_gradient:
            raise ValueError("Compute only additive stochastic gradient is not possible in the logistic setting")
        return self.synthetic_dataset.upper_sigma.dot(w) - y * x

    def sgd_update(self, w, gradient, gamma):
        return minus(w,constant_product(gamma, gradient))

    def gradient_descent(self, label: str = None) -> SGDRun:
        current_w = self.w0
        avg_w = copy.deepcopy(current_w)
        it = 1
        losses = [self.compute_true_risk(current_w, self.X, self.Y)]
        avg_losses = [self.compute_true_risk(avg_w, self.X, self.Y)]
        # matrix_grad = np.zeros((self.SIZE_DATASET, self.DIM))
        for epoch in range(self.NB_EPOCH):
            indices = np.arange(self.SIZE_DATASET)
            for idx in tqdm(indices, disable=DISABLE):
                if idx % MAX_SIZE_DATASET == 0 and idx != 0:
                    print("Regenerating ...")
                    self.synthetic_dataset.regenerate_dataset()
                gamma = self.synthetic_dataset.gamma
                it += 1

                self.Q, self.D = self.synthetic_dataset.Q, self.synthetic_dataset.D

                if self.additive_stochastic_gradient:
                    grad = self.compute_additive_stochastic_gradient(current_w, self.X, self.Y, idx % MAX_SIZE_DATASET)
                else:
                    grad = self.compute_stochastic_gradient(current_w, self.X, self.Y, idx % MAX_SIZE_DATASET)
                g = self.gradient_processing(grad)

                # matrix_grad[idx] = g

                if idx == 0:
                    self.approx_hessian = np.kron(g, g).reshape((self.DIM, self.DIM))
                else:
                    self.approx_hessian = + np.kron(g, g).reshape((self.DIM, self.DIM)) / it + self.approx_hessian * (it - 1)/ it

                self.hessian_processing()

                current_w = self.sgd_update(current_w, g, gamma)
                avg_w = current_w / it + avg_w * (it - 1) / it
                losses.append(self.compute_true_risk(current_w, self.X, self.Y))
                avg_losses.append(self.compute_true_risk(avg_w, self.X, self.Y))

        self.hessian_processing()
        # matrix_cov = matrix_grad.T.dot(matrix_grad) / self.SIZE_DATASET
        print_mem_usage("End of sgd descent ...")
        return SGDRun(current_w, losses, avg_losses, np.diag(self.debiased_hessian), label=label)

    @abstractmethod
    def gradient_processing(self, grad):
        pass

    @abstractmethod
    def hessian_processing(self):
        pass


class SGDVanilla(SGD):

    def gradient_processing(self, grad):
        return grad

    def hessian_processing(self):
        self.debiased_hessian = self.approx_hessian


class SGDNoised(SGD):

    def gradient_processing(self, grad):
        return grad + np.random.normal(0, 1, size=self.DIM)

    def hessian_processing(self):
        self.debiased_hessian = self.approx_hessian


class SGDCompressed(SGD):

    def __init__(self, synthetic_dataset, compressor: CompressionModel) -> None:
        super().__init__(synthetic_dataset)
        self.compressor = compressor
        if isinstance(compressor, RandomSparsification):
            p = compressor.level
            self.inv_proba_matrix = np.eye(self.DIM) - (1 - p) * np.identity(self.DIM)

    # def compute_stochastic_gradient(self, w, data, labels, index):
    #     x, y = data[index], labels[index]
    #     if isinstance(self.compressor, RandomSparsification):
    #         x = self.compressor.compress(x)
    #         p = self.compressor.level
    #         term1 = (1/p) * x * ( (x.dot(w)) / p - y)
    #         term2 = (1-p)/p**2 * (x**2) @ w #np.diag(np.diag(np.array([x]).T @ np.array([x])))
    #         return term1 - term2
    #     return np.array((x @ w - y)).dot(x)

    def hessian_processing(self):
        if isinstance(self.compressor, RandomSparsification):
            self.debiased_hessian = self.approx_hessian * self.inv_proba_matrix
        else:
            self.debiased_hessian = self.approx_hessian

    def gradient_processing(self, grad):
        # if isinstance(self.compressor, RandomSparsification):
        # return grad
        return self.compressor.compress(grad)



