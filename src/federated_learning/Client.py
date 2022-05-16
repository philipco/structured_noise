"""
Created by Constantin Philippenko, 29th April 2022.
"""

import numpy as np
from src.SyntheticDataset import SyntheticDataset

class Client:

    def __init__(self, dim: int, local_size: int, use_ortho_matrix: bool) -> None:
        super().__init__()
        self.dim = dim
        self.local_size = local_size
        self.dataset = SyntheticDataset()
        self.dataset.generate_dataset(dim, size_dataset=local_size, power_cov=2, r_sigma=0,
                                      use_ortho_matrix=use_ortho_matrix, do_logistic_regression=False)
        self.w = self.dataset.w0
        self.avg_w = self.w
        self.local_memory = np.zeros(dim)
        self.alpha = 1 / (2*(self.dataset.quantizator.omega_c + 1))

    def add_to_local_memory(self, update: np.ndarray) -> None:
        self.local_memory += update

    def update_model(self, w: np.ndarray, avg_w: np.ndarray) -> None:
        self.w = w
        self.avg_w = avg_w