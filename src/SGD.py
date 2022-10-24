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

from src.CompressionModel import CompressionModel, SQuantization
from src.JITProduct import *
from src.PickleHandler import pickle_saver
from src.SyntheticDataset import MAX_SIZE_DATASET, SyntheticDataset
from src.Utilities import print_mem_usage
from src.federated_learning.Client import Client

ONLY_ADDITIVE_NOISE = False
USE_MOMENTUM = False
BETA = 0
REGULARIZATION = 0

DISABLE = False
CORRECTION_SQUARE_COV = False
CORRECTION_DIAG = False

IT_THRESHOLD = 0 # 50 for wstar


def log_sampling_xaxix(size_dataset):
    log_len = np.int(math.log10(size_dataset))
    residual_len = math.log10(size_dataset) - log_len
    log_xaxis = [[math.pow(10, a) * math.pow(10, i / 1000) for i in range(1000)] for a in range(log_len)]
    log_xaxis.append([math.pow(10, log_len) * math.pow(10, i / 1000) for i in range(int(1000 * residual_len))])
    log_xaxis = np.concatenate(log_xaxis, axis=None)
    log_xaxis = np.unique(log_xaxis.astype(int))
    return log_xaxis


class SeriesOfSGD:

    def __init__(self) -> None:
        super().__init__()
        self.dict_of_sgd = {}

    def append(self, list_of_sgd):
        for serie in list_of_sgd:
            assert isinstance(serie, SGDRun), "The object added to the series is not of type SGDRun."
            if serie.label in self.dict_of_sgd:
                self.dict_of_sgd[serie.label].append(serie)
            else:
                self.dict_of_sgd[serie.label] = [serie]

    def save(self, filename: str):
        pickle_saver(self, filename)


class SGDRun:

    def __init__(self, size_dataset, nb_epoch: int, sto: bool, batch_size: int, dim:int, last_w, losses, avg_losses,
                 diag_cov_gradients, label=None) -> None:
        super().__init__()
        self.size_dataset = size_dataset
        self.batch_size = batch_size
        self.dim = dim
        self.last_w = last_w
        self.losses = np.array(losses)
        self.avg_losses = np.array(avg_losses)
        if sto:
            self.log_xaxis = log_sampling_xaxix(size_dataset // self.batch_size) * self.batch_size
        else:
            self.log_xaxis = np.arange(nb_epoch)
        self.diag_cov_gradients = diag_cov_gradients
        self.label = label


class SGD(ABC):

    def __init__(self, clients: List[Client], step_formula, nb_epoch: int = 1, sto: bool = True, batch_size: int = 1,
                 reg: int = REGULARIZATION) -> None:
        super().__init__()
        self.clients = clients
        self.sto = sto
        self.batch_size = batch_size
        self.step_formula = step_formula
        if self.sto:
            self.nb_epoch = 1
        else:
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
        self.compressor = SQuantization(0, dim=self.dim)

    def compute_federated_empirical_risk(self, w, avg_w) -> [float, float]:
        # Bien réfléchir au calcul de la loss dans le cas fédéré !!!
        loss = np.mean([self.compute_empirical_risk(w, c.dataset.X_complete, c.dataset.Y, c.dataset.upper_sigma) for c in self.clients])
        avg_loss = np.mean([self.compute_empirical_risk(avg_w, c.dataset.X_complete, c.dataset.Y, c.dataset.upper_sigma) for c in self.clients])
        return loss, avg_loss

    def compute_federated_true_risk(self, w, avg_w) -> [float, float]:
        true_federated_risk = [
            wAw_product(0.5, minus(w, self.clients[i].dataset.w_star), self.clients[i].dataset.upper_sigma)
            - wAw_product(0.5, minus(self.w_star, self.clients[i].dataset.w_star), self.clients[i].dataset.upper_sigma)
                               for i in range(len(self.clients))
        ]
        true_federated_avg_risk = [
                    wAw_product(0.5, minus(avg_w, self.clients[i].dataset.w_star), self.clients[i].dataset.upper_sigma)
                    - wAw_product(0.5, minus(self.w_star, self.clients[i].dataset.w_star), self.clients[i].dataset.upper_sigma)
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

    def compute_stochastic_gradient(self, w, dataset, index, additive_stochastic_gradient):
        if additive_stochastic_gradient:
            return self.compute_additive_stochastic_gradient(w, dataset.X_complete, dataset.Y, index)
        if self.batch_size > 1:
            indices = np.random.choice(min(dataset.size_dataset,MAX_SIZE_DATASET), self.batch_size)
            x, y = dataset.X_complete[indices], dataset.Y[indices]
        else:
            x, y = np.array([dataset.X_complete[index]]), np.array([dataset.Y[index]])

        if CORRECTION_SQUARE_COV:
            x = matrix_vector_product(self.inv_root_square_upper_sigma, x)
        elif CORRECTION_DIAG:
            x = matrix_vector_product(self.Q.T, x)
        return (x @ w - y) @ x / len(y)

    def compute_full_gradient(self, w, dataset):
        return dataset.Xcarre @ w - dataset.Z

    def compute_additive_stochastic_gradient(self, w, data, labels, index):
        x, y = data[index], labels[index]
        if CORRECTION_SQUARE_COV:
            x = matrix_vector_product(self.inv_root_square_upper_sigma, x)
        elif CORRECTION_DIAG:
            x = matrix_vector_product(self.Q.T, x)
        return self.D.dot(w) - y * x

    def sgd_update(self, w, gradient, gamma):
        return w - gamma * gradient #TODO : - self.reg * (w - self.synthetic_dataset.w0)

    def update_approximative_hessian(self, grad, it):
        if it == 0:
            self.approx_hessian = np.kron(grad, grad).reshape((self.dim, self.dim))
        else:
            self.approx_hessian = np.kron(grad, grad).reshape((self.dim, self.dim)) / it + self.approx_hessian * (
                        it - 1) / it

    def gradient_descent(self, label: str = None, deacreasing_step_size: bool = False) -> SGDRun:
        if self.sto:
            log_xaxis = log_sampling_xaxix(self.size_dataset // self.batch_size) * self.batch_size
        else:
            log_xaxis = np.arange(self.nb_epoch)

        current_w = self.w0
        avg_w = copy.deepcopy(current_w)
        it = 1
        current_loss = self.compute_federated_true_risk(current_w, avg_w)
        losses, avg_losses = [current_loss[0]], [current_loss[1]]


        nb_epoch = 1 if self.sto else self.nb_epoch
        for epoch in tqdm(range(nb_epoch), disable=self.sto or DISABLE):

            indices = np.arange(self.size_dataset // self.batch_size) if self.sto else np.array([1])
            for idx in tqdm(indices, disable=not self.sto or DISABLE):

                r2 = np.mean([c.dataset.trace for c in self.clients])
                gamma = self.step_formula(it, r2, self.compressor.omega_c)
                grad = np.zeros(self.dim)

                for client in self.clients:
                    if idx % (MAX_SIZE_DATASET // self.batch_size) == 0 and idx != 0:
                        print("Regenerating ...")
                        client.dataset.regenerate_dataset()

                    local_grad = self.compute_gradient(
                        client.w, client.dataset, idx % (MAX_SIZE_DATASET // self.batch_size),
                        self.additive_stochastic_gradient)
                    # Smart initialization
                    # if it == 1:
                    #     client.local_memory = local_grad
                    grad += self.gradient_processing(local_grad, client)

                grad /= len(self.clients)

                self.update_approximative_hessian(grad, it)
                it += 1

                current_w = self.sgd_update(current_w, grad, gamma)
                if it <= IT_THRESHOLD:
                    avg_w = current_w
                else:
                    avg_w = current_w / (it-IT_THRESHOLD) + avg_w * (it - IT_THRESHOLD- 1) / (it - IT_THRESHOLD)

                for client in self.clients:
                    client.update_model(current_w, avg_w)

                current_loss = self.compute_federated_true_risk(current_w, avg_w)
                if not self.sto and epoch in log_xaxis[1:]:
                    losses.append(current_loss[0])
                    avg_losses.append(current_loss[1])
                elif self.sto and idx * self.batch_size in log_xaxis[1:]:
                    losses.append(current_loss[0])
                    avg_losses.append(current_loss[1])

        print_mem_usage("End of sgd descent ...")

        if self.use_ortho_matrix:
            cov_matrix = self.ortho_matrix.T.dot(self.approx_hessian).dot(self.ortho_matrix)
        else:
            cov_matrix = self.approx_hessian

        print("Local gradient at optimal point:")
        optimal_grad = [np.linalg.norm(self.compute_full_gradient(avg_w, c.dataset)) for c in self.clients]
        print("B^2 = ", np.mean(optimal_grad))
        return SGDRun(self.size_dataset, self.nb_epoch, self.sto, self.batch_size, self.dim, avg_w, losses, avg_losses,
                      np.diag(cov_matrix), label=label)

    @abstractmethod
    def gradient_processing(self, grad, client: Client):
        pass

    @abstractmethod
    def compute_gradient(client, w, dataset: SyntheticDataset, idx, additive_stochastic_gradient):
        pass


class FullGD(SGD):

    def compute_gradient(self, w, dataset: SyntheticDataset, idx, additive_stochastic_gradient):
        return self.compute_full_gradient(w, dataset)

    def gradient_processing(self, grad, client: Client):
        return grad


class SGDVanilla(SGD):

    def compute_gradient(self, w, dataset: SyntheticDataset, idx, additive_stochastic_gradient):
        if self.sto:
            return self.compute_stochastic_gradient(w, dataset, idx, additive_stochastic_gradient)
        else:
            return self.compute_full_gradient(w, dataset)

    def gradient_processing(self, grad, client: Client):
        return grad


class SGDNoised(SGD):

    def compute_gradient(self, w, dataset: SyntheticDataset, idx):
        return self.compute_stochastic_gradient(w, dataset, idx, self.additive_stochastic_gradient)

    def gradient_processing(self, grad, client: Client):
        return grad + np.random.normal(0, 1, size=self.dim)


class SGDCompressed(SGD):

    def __init__(self, clients: List[Client], step_formula, compressor: CompressionModel, nb_epoch: int = 1, sto: bool = True,
                 batch_size: int = 1) -> None:
        super().__init__(clients, step_formula, nb_epoch, sto, batch_size)
        self.compressor = compressor

    def compute_gradient(self, w, dataset: SyntheticDataset, idx, additive_stochastic_gradient):
        if self.sto:
            return self.compute_stochastic_gradient(w, dataset, idx, additive_stochastic_gradient)
        else:
            return self.compute_full_gradient(w, dataset)

    def gradient_processing(self, grad, client: Client):
        return self.compressor.compress(grad)


class SGDArtemis(SGD):

    def __init__(self, clients: List[Client], step_formula, compressor: CompressionModel, nb_epoch: int = 1, sto: bool = True,
                 batch_size: int = 1) -> None:
        super().__init__(clients, step_formula, nb_epoch, sto, batch_size)
        self.compressor = compressor

    def compute_gradient(self, w, dataset: SyntheticDataset, idx, additive_stochastic_gradient):
        if self.sto:
            return self.compute_stochastic_gradient(w, dataset, idx, additive_stochastic_gradient)
        else:
            return self.compute_full_gradient(w, dataset)

    def gradient_processing(self, grad, client: Client):
        compressed = self.compressor.compress(grad - client.local_memory)
        compressed_grad = compressed + client.local_memory
        alpha = 1 / (2*(self.compressor.omega_c + 1))
        client.local_memory += alpha * compressed
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

