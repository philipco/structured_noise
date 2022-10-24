"""
Created by Constantin Philippenko, 29th April 2022.
"""
from typing import List

import numpy as np
from src.SyntheticDataset import SyntheticDataset


class Client:

    def __init__(self, client_id: int, dim: int, local_size: int, power_cov: int, nb_clients: int, use_ortho_matrix: bool,
                 heterogeneity: str, eigenvalues = None, w0_seed: int = 42) -> None:
        super().__init__()
        self.client_id = client_id
        self.dim = dim
        self.local_size = local_size
        self.dataset = SyntheticDataset()
        self.dataset.generate_dataset(dim, size_dataset=local_size, power_cov=power_cov, r_sigma=0,
                                      nb_clients=nb_clients, use_ortho_matrix=use_ortho_matrix,
                                      do_logistic_regression=False, heterogeneity=heterogeneity,
                                      eigenvalues=eigenvalues, w0_seed=w0_seed, client_id=client_id)
        self.w = self.dataset.w0
        self.avg_w = self.w
        self.local_memory = np.zeros(dim)

    def regenerate_dataset(self):
        self.dataset.generate_X()
        self.dataset.generate_Y()
        self.w = self.dataset.w0
        self.avg_w = self.w
        self.local_memory = np.zeros(self.dim)

    def update_model(self, w: np.ndarray, avg_w: np.ndarray) -> None:
        self.w = w
        self.avg_w = avg_w

def check_clients(clients: List[Client], heterogeneity: str):
    for c in clients[1:]:
        assert (clients[0].dataset.w0 == c.dataset.w0).any()
        if heterogeneity == "wstar":
            assert (clients[0].dataset.w_star != c.dataset.w_star).any()
            assert (clients[0].dataset.ortho_matrix == c.dataset.ortho_matrix).any()
            assert (clients[0].dataset.upper_sigma == c.dataset.upper_sigma).any()
        elif heterogeneity == "sigma":
            assert (clients[0].dataset.w_star == c.dataset.w_star).any()
            assert (clients[0].dataset.upper_sigma != c.dataset.upper_sigma).any()
        elif heterogeneity == "homog":
            assert (clients[0].dataset.w_star == c.dataset.w_star).any()
            assert (clients[0].dataset.ortho_matrix == c.dataset.ortho_matrix).any()
            assert (clients[0].dataset.upper_sigma == c.dataset.upper_sigma).any()
        else:
            raise ValueError("Parameter of heterogeneity is unkrecognized.")