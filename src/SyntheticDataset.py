"""
Created by Constantin Philippenko, 10th January 2022.
"""
import sys

import numpy as np
from numpy.random import multivariate_normal
from scipy.stats import ortho_group

from src.CompressionModel import SQuantization, RandomSparsification


class SyntheticDataset:

    def __init__(self):#, dim: int, size_dataset: int, power_cov: int, use_ortho_matrix: bool) -> None:
        super().__init__()

    def generate_dataset(self, dim: int, size_dataset: int, power_cov: int, use_ortho_matrix: bool):
        self.generate_X(dim, size_dataset, power_cov, use_ortho_matrix)
        self.generate_Y()
        self.set_step_size()

    def generate_X(self, dim: int, size_dataset: int, power_cov: int, use_ortho_matrix: bool):
        self.dim = dim
        self.power_cov = power_cov

        # Used to generate self.X
        self.upper_sigma = np.diag(np.array([1 / (i ** power_cov) for i in range(1, dim + 1)]), k=0)
        if use_ortho_matrix:
            self.ortho_matrix = ortho_group.rvs(dim=self.dim)
        else:
            self.ortho_matrix = np.diag(np.array([1 for i in range(1, dim + 1)]), k=0)

        self.size_dataset = size_dataset

        self.X = multivariate_normal(np.zeros(self.dim), self.upper_sigma, size=self.size_dataset)
        self.X = self.X.dot(self.ortho_matrix.T)

        print("Memory footprint X", sys.getsizeof(self.X))
        print("Memory footprint SIGMA", sys.getsizeof(self.upper_sigma))

    def generate_Y(self):
        lower_sigma = 1  # Used only to introduce noise in the true labels.
        self.w_star = np.ones(self.dim) #np.power(self.upper_sigma, 1/4) @
        self.Y = self.X @ self.w_star + np.random.normal(0, lower_sigma, size=self.size_dataset)

    def string_for_hash(self):
        return "{0}{1}{2}{3}".format(self.size_dataset, self.dim, self.power_cov, self.GAMMA)

    def set_step_size(self):
        EIGEN_VALUES, _ = np.linalg.eig(self.X.T @ self.X)
        self.L = np.max(EIGEN_VALUES) / self.size_dataset
        print("L=", self.L)

        R_SQUARE = np.trace(self.upper_sigma)
        print("4 * R_SQUARE", 4 * R_SQUARE)

        GAMMA_BACH_MOULINES = 1 / (4 * R_SQUARE)

        TARGET_OMEGA = 1 # 10 if dim=500, else = 4
        self.LEVEL_QTZ = 1 # np.floor(np.sqrt(self.dim) / TARGET_OMEGA)  # Lead to omega_c = 3.
        self.quantizator = SQuantization(self.LEVEL_QTZ, dim=self.dim)

        self.LEVEL_RDK = 1 / (self.quantizator.omega_c + 1)
        self.sparsificator = RandomSparsification(self.LEVEL_RDK, dim=self.dim, biased=False)

        print("Level qtz:", self.LEVEL_QTZ)
        print("Level rdk:", self.LEVEL_RDK)
        print("Qtz compression:", self.quantizator.omega_c)
        print("Rdk compression:", self.sparsificator.omega_c)

        OPTIMAL_GAMMA_COMPR = 1 / (self.L * (1 + 2 * (SQuantization(self.LEVEL_QTZ, dim=self.dim).omega_c + 1)))
        print("Optimal gamma for compression:", OPTIMAL_GAMMA_COMPR)
        print("Gamma from Bach & Moulines, 13:", GAMMA_BACH_MOULINES)

        CONSTANT_GAMMA = .1 / (2 * self.L)
        print("Constant step size:", CONSTANT_GAMMA)

        self.GAMMA = OPTIMAL_GAMMA_COMPR

        print("Take step size:", self.GAMMA)

