"""
Created by Constantin Philippenko, 10th January 2022.
"""

import copy
import math
from abc import abstractmethod, ABC
from typing import List

import numpy as np
from numpy.linalg import inv
from tqdm import tqdm

from src.CompressionModel import CompressionModel, Quantization
from src.JITProduct import *
from src.SyntheticDataset import MAX_SIZE_DATASET, SyntheticDataset, AbstractDataset
from src.federated_learning.Client import Client, ClientRealDataset
from src.utilities.PickleHandler import pickle_saver
from src.utilities.Utilities import print_mem_usage

ONLY_ADDITIVE_NOISE = False
USE_MOMENTUM = False
BETA = 0
REGULARIZATION = 0

DISABLE = False

def log_sampling_xaxix(size_dataset: int, nb_epoch: int = 1) -> np.ndarray:
    size_dataset *= nb_epoch
    log_len = np.int(math.log10(size_dataset))
    residual_len = math.log10(size_dataset) - log_len
    log_xaxis = [[math.pow(10, a) * math.pow(10, i / 1000) for i in range(1000)] for a in range(log_len)]
    log_xaxis.append([math.pow(10, log_len) * math.pow(10, i / 1000) for i in range(int(1000 * residual_len))])
    log_xaxis = np.concatenate(log_xaxis, axis=None)
    log_xaxis = np.unique(log_xaxis.astype(int))
    return log_xaxis


def compute_wstar(clients: ClientRealDataset, batch_size: int) -> float:
    print(">>>>> Computing w_star.")
    # We temporaly set w_star to zero in order to run the SGD.
    for c in clients:
        c.dataset.w_star = np.zeros(c.dataset.dim)

    vanilla_sgd = SGDVanilla(clients, lambda it, r2, omega, K: 1 / (2 * (omega + 1) * r2), sto=True, batch_size=batch_size,
                             nb_epoch=200*len(clients))
    sgd_nocompr = vanilla_sgd.gradient_descent(label="wstar")
    return sgd_nocompr.last_w


class SGDRun:

    def __init__(self, size_dataset, nb_epoch: int, sto: bool, batch_size: int, dim:int, last_w, losses, avg_losses,
                 label=None) -> None:
        super().__init__()
        self.size_dataset = size_dataset
        self.batch_size = batch_size
        self.dim = dim
        self.last_w = last_w
        self.losses = np.array(losses)
        self.avg_losses = np.array(avg_losses)
        self.nb_epoch = nb_epoch
        # if sto:
        self.log_xaxis = log_sampling_xaxix((size_dataset) // self.batch_size,  self.nb_epoch) * self.batch_size
        # else:
        #     self.log_xaxis = np.arange(nb_epoch)
        self.label = label


class SGD(ABC):

    def __init__(self, clients: List[Client], step_formula, nb_epoch: int = 1, sto: bool = True, batch_size: int = 1,
                 start_averaging: int = 0, reg: int = 0) -> None:
        super().__init__()
        self.clients = clients
        self.sto = sto
        self.batch_size = batch_size
        self.step_formula = step_formula

        self.nb_epoch = nb_epoch

        self.start_averaging = start_averaging
        self.L, self.dim, self.gamma = np.mean([c.dataset.L for c in clients]), clients[0].dim, 0.1
        self.size_dataset, self.w0 = clients[0].local_size, clients[0].dataset.w0
        self.sigma = clients[0].dataset.upper_sigma

        self.additive_stochastic_gradient = ONLY_ADDITIVE_NOISE

        if not clients[0].dataset.real_dataset:
            self.use_ortho_matrix = clients[0].dataset.use_ortho_matrix
            self.ortho_matrix = clients[0].dataset.ortho_matrix

        self.w_star = np.mean([client.dataset.w_star for client in self.clients], axis=0)

        self.Q, self.D = np.identity(self.dim), np.identity(self.dim)
        self.reg = reg
        self.compressor = Quantization(0, dim=self.dim)

    def compute_federated_true_risk(self, w: np.ndarray, avg_w: np.ndarray) -> [float, float]:
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

    def compute_stochastic_gradient(self, w: np.ndarray, dataset: AbstractDataset, index: int,
                                    additive_stochastic_gradient: bool) -> np.ndarray:
        if additive_stochastic_gradient:
            return self.compute_additive_stochastic_gradient(w, dataset.X, dataset.Y, index)
        if self.batch_size > 1:
            indices = np.random.choice(min(dataset.size_dataset, MAX_SIZE_DATASET), self.batch_size)
            x, y = dataset.X[indices], dataset.Y[indices]
        else:
            x, y = np.array([dataset.X[index]]), np.array([dataset.Y[index]])

        return (x @ w - y) @ x / len(y)

    def compute_full_gradient(self, w: np.ndarray, dataset: AbstractDataset) -> np.ndarray:
        return dataset.upper_sigma @ (w - dataset.w_star)

    def compute_additive_stochastic_gradient(self, w: np.ndarray, data: np.ndarray, labels: np.ndarray, index: int) \
            -> np.ndarray:
        x, y = data[index], labels[index]
        return self.D.dot(w) - y * x

    def sgd_update(self, w: np.ndarray, gradient: np.ndarray, gamma: float) -> np.ndarray:
        return w - gamma * (gradient + self.reg * (w - self.w0))

    def gradient_descent(self, label: str = None) -> SGDRun:
        # if self.sto:
        log_xaxis = log_sampling_xaxix(self.size_dataset // self.batch_size, self.nb_epoch) * self.batch_size
        # else:
        #     log_xaxis = np.arange(self.nb_epoch)

        current_w = self.w0
        avg_w = copy.deepcopy(current_w)
        it = 1
        current_loss = self.compute_federated_true_risk(current_w, avg_w)
        losses, avg_losses = [current_loss[0]], [current_loss[1]]

        indices = np.arange((self.size_dataset * self.nb_epoch) // self.batch_size) #if self.sto else np.array([1])
        for idx in tqdm(indices, disable=not self.sto or DISABLE):

            r2 = np.mean([c.dataset.trace for c in self.clients])
            gamma = self.step_formula(it, r2, self.compressor.omega_c, len(indices)*self.batch_size)
            grad = np.zeros(self.dim)

            for client in self.clients:
                if idx % (MAX_SIZE_DATASET // self.batch_size) == 0 and idx != 0:
                    print("Regenerating ...")
                    client.dataset.regenerate_dataset()

                local_grad = self.compute_gradient(
                    client.w, client.dataset, idx % (min(MAX_SIZE_DATASET, self.size_dataset) // self.batch_size),
                    self.additive_stochastic_gradient)
                grad += self.gradient_processing(local_grad, client)

            grad /= len(self.clients)

            it += 1

            current_w = self.sgd_update(current_w, grad, gamma)
            if it <= self.start_averaging:
                avg_w = current_w
            else:
                avg_w = current_w / (it-self.start_averaging) \
                        + avg_w * (it - self.start_averaging- 1) / (it - self.start_averaging)

            for client in self.clients:
                client.update_model(current_w, avg_w)

            current_loss = self.compute_federated_true_risk(current_w, avg_w)
            if idx * self.batch_size in log_xaxis[1:]:
                losses.append(current_loss[0])
                avg_losses.append(current_loss[1])

        print_mem_usage("End of sgd descent ...")
        return SGDRun(self.size_dataset, self.nb_epoch, self.sto, self.batch_size, self.dim, avg_w, losses, avg_losses,
                      label=label)

    @abstractmethod
    def gradient_processing(self, grad: np.ndarray, client: Client) -> np.ndarray:
        pass

    @abstractmethod
    def compute_gradient(client, w: np.ndarray, dataset: AbstractDataset, idx: int, additive_stochastic_gradient: bool)\
            -> np.ndarray:
        pass


class FullGD(SGD):

    def compute_gradient(self, w: np.ndarray, dataset: AbstractDataset, idx: int, additive_stochastic_gradient: bool) \
            -> np.ndarray:
        return self.compute_full_gradient(w, dataset)

    def gradient_processing(self, grad: np.ndarray, client: Client) -> np.ndarray:
        return grad


class SGDVanilla(SGD):

    def compute_gradient(self, w: np.ndarray, dataset: AbstractDataset, idx: int, additive_stochastic_gradient: bool) \
            -> np.ndarray:
        if self.sto:
            return self.compute_stochastic_gradient(w, dataset, idx, additive_stochastic_gradient)
        else:
            return self.compute_full_gradient(w, dataset)

    def gradient_processing(self, grad: np.ndarray, client: Client) -> np.ndarray:
        return grad


class SGDCompressed(SGD):

    def __init__(self, clients: List[Client], step_formula, compressor: CompressionModel, nb_epoch: int = 1,
                 sto: bool = True, batch_size: int = 1, reg: int = 0, start_averaging: int = 0,) -> None:
        super().__init__(clients, step_formula, nb_epoch, sto, batch_size, start_averaging, reg)
        self.compressor = compressor

    def compute_gradient(self, w: np.ndarray, dataset: AbstractDataset, idx: int, additive_stochastic_gradient: bool) \
            -> np.ndarray:
        if self.sto:
            return self.compute_stochastic_gradient(w, dataset, idx, additive_stochastic_gradient)
        else:
            return self.compute_full_gradient(w, dataset)

    def gradient_processing(self, grad: np.ndarray, client: Client) -> np.ndarray:
        return self.compressor.compress(grad)


class SGDArtemis(SGD):

    def __init__(self, clients: List[Client], step_formula, compressor: CompressionModel, nb_epoch: int = 1,
                 sto: bool = True, batch_size: int = 1, reg: int = 0, start_averaging: int = 0) -> None:
        super().__init__(clients, step_formula, nb_epoch, sto, batch_size, reg)
        self.compressor = compressor

    def compute_gradient(self, w: np.ndarray, dataset: AbstractDataset, idx: int, additive_stochastic_gradient: bool) \
            -> np.ndarray:
        if self.sto:
            return self.compute_stochastic_gradient(w, dataset, idx, additive_stochastic_gradient)
        else:
            return self.compute_full_gradient(w, dataset)

    def gradient_processing(self, grad: np.ndarray, client: Client) -> np.ndarray:
        compressed = self.compressor.compress(grad - client.local_memory)
        compressed_grad = compressed + client.local_memory
        alpha = 1 / (2*(self.compressor.omega_c + 1))
        client.local_memory += alpha * compressed
        return compressed_grad


class SeriesOfSGD:

    def __init__(self) -> None:
        super().__init__()
        self.dict_of_sgd = {}

    def append(self, list_of_sgd: List[SGDRun]) -> None:
        for serie in list_of_sgd:
            assert isinstance(serie, SGDRun), "The object added to the series is not of type SGDRun."
            if serie.label in self.dict_of_sgd:
                self.dict_of_sgd[serie.label].append(serie)
            else:
                self.dict_of_sgd[serie.label] = [serie]

    def save(self, filename: str) -> None:
        pickle_saver(self, filename)