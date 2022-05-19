"""
Created by Constantin Philippenko, 10th January 2022.
"""

import copy
import math
from abc import abstractmethod, ABC
from typing import List

from numpy.linalg import inv
from scipy.linalg import sqrtm
from tqdm import tqdm

from src.CompressionModel import CompressionModel
from src.JITProduct import *
from src.PickleHandler import pickle_saver
from src.SyntheticDataset import MAX_SIZE_DATASET
from src.Utilities import print_mem_usage
from src.federated_learning.Client import Client

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

    def __init__(self, list_of_sgd) -> None:
        super().__init__()
        self.series = []
        for serie in list_of_sgd:
            assert isinstance(serie, SGDRun), "The object added to the series is not of type SGDRun."
            self.series.append(serie)

    def append(self, list_of_sgd):
        for serie in list_of_sgd:
            assert isinstance(serie, SGDRun), "The object added to the series is not of type SGDRun."
            self.series.append(serie)

    def save(self, filename: str):
        pickle_saver(self, filename)


class SGDRun:

    def __init__(self, size_dataset, all_avg_w, last_w, losses, avg_losses, diag_cov_gradients, label=None) -> None:
        super().__init__()
        self.size_dataset = size_dataset
        self.all_avg_w = all_avg_w
        self.last_w = last_w
        self.losses = np.array(losses)
        self.avg_losses = np.array(avg_losses)
        self.log_xaxis = log_sampling_xaxix(size_dataset)
        self.diag_cov_gradients = diag_cov_gradients
        self.label = label


class SGD(ABC):
    
    def __init__(self, clients: List[Client], nb_epoch: int = 1, reg: int = REGULARIZATION) -> None:
        super().__init__()
        self.clients = clients
        self.nb_epoch = nb_epoch
        self. L, self.dim, self.gamma = np.mean([c.dataset.L for c in clients]), clients[0].dim, 0.1
        self.size_dataset, self.w0 = clients[0].local_size, clients[0].dataset.w0
        self.w_star = np.mean([client.dataset.w_star for client in self.clients], axis=0)
        self.sigma = clients[0].dataset.upper_sigma

        self.additive_stochastic_gradient = ONLY_ADDITIVE_NOISE

        self.use_ortho_matrix = clients[0].dataset.use_ortho_matrix
        self.ortho_matrix = clients[0].dataset.ortho_matrix
        self.root_square_upper_sigma = sqrtm(self.sigma)
        self.inv_root_square_upper_sigma = inv(self.root_square_upper_sigma)

        self.Q, self.D = np.identity(self.dim), np.identity(self.dim)
        self.approx_hessian = np.identity(self.dim)
        self.debiased_hessian = np.identity(self.dim)
        self.reg = reg
        self.compressor = None

    def compute_federated_empirical_risk(self, w, avg_w) -> [float, float]:
        # Bien réfléchir au calcul de la loss dans le cas fédéré !!!
        loss = np.mean([self.compute_empirical_risk(w, c.dataset.X_complete, c.dataset.Y, c.dataset.upper_sigma) for c in self.clients])
        avg_loss = np.mean([self.compute_empirical_risk(avg_w, c.dataset.X_complete, c.dataset.Y, c.dataset.upper_sigma) for c in self.clients])
        return loss, avg_loss

    def compute_federated_true_risk(self, w, avg_w, sigma) -> [float, float]:
        true_federated_risk = [
            wAw_product(0.5, minus(w, self.clients[i].dataset.w_star), sigma)
            - wAw_product(0.5, minus(self.w_star, self.clients[i].dataset.w_star), sigma)
                               for i in range(len(self.clients))
        ]
        true_federated_avg_risk = [
                    wAw_product(0.5, minus(avg_w, self.clients[i].dataset.w_star), sigma)
                    - wAw_product(0.5, minus(self.w_star, self.clients[i].dataset.w_star), sigma)
                                       for i in range(len(self.clients))
                ]
        return np.mean(true_federated_risk, axis=0), np.mean(true_federated_avg_risk, axis=0)

    def compute_empirical_risk(self, w, data, labels, sigma):
        if CORRECTION_SQUARE_COV:
            data = data @ self.inv_root_square_upper_sigma.T
        if CORRECTION_DIAG:
            data = data @ self.Q
        return 0.5 * np.linalg.norm(data @ w - labels) ** 2 / len(labels)

    def compute_true_risk(self, w, data, labels, sigma):
        if data is None:
            return 0
        if CORRECTION_SQUARE_COV:
            w_star = self.inv_root_square_upper_sigma @ self.w_star
            return constant_product(0.5, vectorial_norm(minus(w, w_star)))
        elif CORRECTION_DIAG:
            w_star = matrix_vector_product(self.Q.T, self.w_star)
            return wAw_product(0.5, minus(w, w_star), self.D)
        return wAw_product(0.5, minus(w, self.w_star), sigma)

    def compute_stochastic_gradient(self, w, data, labels, index, additive_stochastic_gradient):
        # index = np.random.randint(len(labels))
        if additive_stochastic_gradient:
            return self.compute_additive_stochastic_gradient(w, data, labels, index)
        x, y = data[index], labels[index]
        if CORRECTION_SQUARE_COV:
            x = matrix_vector_product(self.inv_root_square_upper_sigma, x)
        elif CORRECTION_DIAG:
            x = matrix_vector_product(self.Q.T, x)
        return (x @ w - y) * x

    def compute_full_gradient(self, w, data, labels):
        return (data @ w - labels) @ data / len(labels)

    def compute_additive_stochastic_gradient(self, w, data, labels, index):
        x, y = data[index], labels[index]
        if CORRECTION_SQUARE_COV:
            x = matrix_vector_product(self.inv_root_square_upper_sigma, x)
        elif CORRECTION_DIAG:
            x = matrix_vector_product(self.Q.T, x)
        return self.D.dot(w) - y * x

    def sgd_update(self, w, gradient, gamma):
        return w - gamma * gradient #TODO : - self.reg * (w - self.synthetic_dataset.w0)

    def get_step_size(self, it: int, gamma: int, deacreasing_step_size: bool = False):
        if deacreasing_step_size:
            return 1 / (np.sqrt(it) * self. L)
        return 1 / (2 * self.L ) #* (1 + 2 * (self.clients[0].dataset.quantizator.omega_c + 1)))

    def update_approximative_hessian(self, grad, it):
        if it == 0:
            self.approx_hessian = np.kron(grad, grad).reshape((self.dim, self.dim))
        else:
            self.approx_hessian = np.kron(grad, grad).reshape((self.dim, self.dim)) / it + self.approx_hessian * (
                        it - 1) / it

    def gradient_descent(self, label: str = None, deacreasing_step_size: bool = False) -> SGDRun:
        log_xaxis = log_sampling_xaxix(self.size_dataset)

        all_avg_w = []
        current_w = self.w0
        avg_w = copy.deepcopy(current_w)
        it = 1
        current_loss = self.compute_federated_empirical_risk(current_w, avg_w)
        losses, avg_losses = [current_loss[0]], [current_loss[1]]

        for epoch in range(self.nb_epoch):
            indices = np.arange(self.size_dataset)
            for idx in tqdm(indices, disable=DISABLE):

                gamma = self.get_step_size(it, self.gamma, deacreasing_step_size)
                grad = np.zeros(self.dim)

                for client in self.clients:
                    if idx % MAX_SIZE_DATASET == 0 and idx != 0:
                        print("Regenerating ...")
                        client.dataset.regenerate_dataset()

                    local_grad = self.compute_gradient(
                        client.w, client.dataset.X_complete, client.dataset.Y, idx % MAX_SIZE_DATASET,
                        self.additive_stochastic_gradient)
                    if it == 1:
                        client.local_memory = local_grad
                    grad += self.gradient_processing(local_grad, client)

                grad /= len(self.clients)

                self.update_approximative_hessian(grad, it)
                it += 1

                current_w = self.sgd_update(current_w, grad, gamma)
                avg_w = current_w / it + avg_w * (it - 1) / it
                all_avg_w.append(avg_w)

                for client in self.clients:
                    client.update_model(current_w, avg_w)

                current_loss = self.compute_federated_true_risk(current_w, avg_w, self.sigma)
                if idx in log_xaxis[1:]:
                    losses.append(current_loss[0])
                    avg_losses.append(current_loss[1])

        print_mem_usage("End of sgd descent ...")

        if self.use_ortho_matrix:
            cov_matrix = self.ortho_matrix.T.dot(self.approx_hessian).dot(self.ortho_matrix)
        else:
            cov_matrix = self.approx_hessian

        return SGDRun(self.size_dataset, all_avg_w, current_w, losses, avg_losses, np.diag(cov_matrix),
                      label=label)

    @abstractmethod
    def gradient_processing(self, grad, client: Client):
        pass

    @abstractmethod
    def compute_gradient(client, w, X, Y, idx, additive_stochastic_gradient):
        pass


class FullGD(SGD):

    def compute_gradient(self, w, X, Y, idx, additive_stochastic_gradient):
        return self.compute_full_gradient(w, X, Y)

    def gradient_processing(self, grad, client: Client):
        return grad


class SGDVanilla(SGD):

    def compute_gradient(self, w, X, Y, idx, additive_stochastic_gradient):
        return self.compute_stochastic_gradient(w, X, Y, idx, additive_stochastic_gradient)

    def gradient_processing(self, grad, client: Client):
        return grad


class SGDNoised(SGD):

    def compute_gradient(self, w, X, Y, idx):
        return self.compute_stochastic_gradient(w, X, Y, idx, self.additive_stochastic_gradient)

    def gradient_processing(self, grad, client: Client):
        return grad + np.random.normal(0, 1, size=self.dim)


class SGDCompressed(SGD):

    def __init__(self, clients: List[Client], compressor: CompressionModel, nb_epoch: int = 1) -> None:
        super().__init__(clients, nb_epoch)
        self.compressor = compressor

    def compute_gradient(self, w, X, Y, idx, additive_stochastic_gradient):
        return self.compute_stochastic_gradient(w, X, Y, idx, additive_stochastic_gradient)

    def gradient_processing(self, grad, client: Client):
        return self.compressor.compress(grad)


class SGDArtemis(SGD):

    def __init__(self, clients: List[Client], compressor: CompressionModel, nb_epoch: int = 1) -> None:
        super().__init__(clients, nb_epoch)
        self.compressor = compressor

    def compute_gradient(self, w, X, Y, idx, additive_stochastic_gradient):
        return self.compute_stochastic_gradient(w, X, Y, idx, additive_stochastic_gradient)

    def gradient_processing(self, grad, client: Client):
        compressed = self.compressor.compress(grad - client.local_memory)
        compressed_grad = compressed + client.local_memory
        client.local_memory += client.alpha * compressed
        return compressed_grad


class SGDSportisse(SGD):

    def __init__(self, synthetic_dataset, compressor: CompressionModel) -> None:
        super().__init__(synthetic_dataset)
        self.compressor = compressor

    def gradient_processing(self, grad, client: Client):
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

    def gradient_processing(self, grad, client: Client):
        return grad

    def compute_stochastic_gradient(self, w, data, labels, index):
        x, y = data[index], labels[index]
        x = self.synthetic_dataset.sparsificator.compress(x)
        g = x * (w @ x - y)
        return g

