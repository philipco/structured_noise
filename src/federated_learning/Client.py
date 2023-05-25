"""
Created by Constantin Philippenko, 09th August 2022.
"""
from typing import List

import numpy as np

from src import RealDataset
from src.SyntheticDataset import SyntheticDataset


class Client:

    def __init__(self, client_id: int, dim: int, local_size: int, power_cov: int, nb_clients: int, use_ortho_matrix: bool,
                 heterogeneity: str, eigenvalues = None, w0_seed: int = 42, lower_sigma: int = None) -> None:
        super().__init__()
        self.client_id = client_id
        self.dim = dim
        self.local_size = local_size
        self.dataset = SyntheticDataset()
        self.dataset.generate_dataset(dim, size_dataset=local_size, power_cov=power_cov, r_sigma=0,
                                      nb_clients=nb_clients, use_ortho_matrix=use_ortho_matrix,
                                      do_logistic_regression=False, heterogeneity=heterogeneity,
                                      eigenvalues=eigenvalues, w0_seed=w0_seed, client_id=client_id,
                                      lower_sigma=lower_sigma)
        self.w = self.dataset.w0
        self.avg_w = self.w
        self.local_memory = np.zeros(dim)

    def regenerate_dataset(self) -> None:
        self.dataset.generate_X()
        self.dataset.generate_Y()
        self.w = self.dataset.w0
        self.avg_w = self.w
        self.local_memory = np.zeros(self.dim)

    def update_model(self, w: np.ndarray, avg_w: np.ndarray) -> None:
        self.w = w
        self.avg_w = avg_w


class ClientRealDataset:

    def __init__(self, client_id: int, dim: int, local_size: int, dataset: RealDataset) -> None:
        super().__init__()
        self.client_id = client_id
        self.dim = dim
        self.local_size = local_size
        self.dataset = dataset
        self.w = self.dataset.w0
        self.avg_w = self.w
        self.local_memory = np.zeros(dim)
        self.regenerate_dataset()  # We shuffle data also at initialisation.

    def update_model(self, w: np.ndarray, avg_w: np.ndarray) -> None:
        self.w = w
        self.avg_w = avg_w

    def regenerate_dataset(self) -> None:
        # Concatenate X_complete and Y along the last axis
        data = np.concatenate((self.dataset.X, self.dataset.Y.reshape(-1, 1)), axis=-1)

        # Shuffle the data
        np.random.shuffle(data)

        # Split X_complete and Y again
        self.dataset.X = data[:, :-1]
        self.dataset.Y = data[:, -1].astype(np.int64)


def check_clients(clients: List[Client], heterogeneity: str) -> None:
    for c in clients[1:]:
        # assert (clients[0].dataset.w0 == c.dataset.w0).any()
        if heterogeneity == "wstar":
            assert (clients[0].dataset.w_star != c.dataset.w_star).any()
            assert (clients[0].dataset.ortho_matrix == c.dataset.ortho_matrix).any()
            assert (clients[0].dataset.upper_sigma == c.dataset.upper_sigma).any()
        elif heterogeneity == "sigma":
            assert (clients[0].dataset.w_star == c.dataset.w_star).any()
        elif heterogeneity == "homog":
            assert (clients[0].dataset.w_star == c.dataset.w_star).any()
            assert (clients[0].dataset.ortho_matrix == c.dataset.ortho_matrix).any()
            assert (clients[0].dataset.upper_sigma == c.dataset.upper_sigma).any()
        else:
            raise ValueError("Parameter of heterogeneity is unkrecognized.")
    # We check that at least one covariance is different to the one of the first client.
    if heterogeneity == "sigma":
        any([clients[0].dataset.power_cov != c.dataset.power_cov for c in clients[1:]])
