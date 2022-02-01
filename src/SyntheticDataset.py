"""
Created by Constantin Philippenko, 10th January 2022.
"""
import math
import sys

import numpy as np
from matplotlib import pyplot as plt
from numpy.random import multivariate_normal
from scipy.special import expit
from scipy.stats import ortho_group

from src.CompressionModel import SQuantization, RandomSparsification

MAX_SIZE_DATASET = 10**7


class AbstractDataset:

    def __init__(self, name: str = None) -> None:
        super().__init__()
        self.name = name

    def string_for_hash(self):
        if self.name:
            return "{0}-N{1}-D{2}-P{3}-R{4}".format(self.name, self.size_dataset, self.dim, self.power_cov, self.r_sigma)
        return "N{0}-D{1}-P{2}-R{3}".format(self.size_dataset, self.dim, self.power_cov, self.r_sigma)

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

        self.gamma = OPTIMAL_GAMMA_COMPR

        print("Take step size:", self.gamma)


class RealLifeDataset(AbstractDataset):

    def load_data(self, X, Y, do_logistic_regression: bool, name: str = None):
        self.do_logistic_regression = do_logistic_regression
        self.X, self.Y = X, Y
        self.upper_sigma = self.X.T @ self.X
        self.w_star = None
        self.size_dataset, self.dim = X.shape[0], X.shape[1]
        self.set_step_size()


class SyntheticDataset(AbstractDataset):

    def generate_dataset(self, dim: int, size_dataset: int, power_cov: int, r_sigma: int, use_ortho_matrix: bool,
                         do_logistic_regression: bool):
        self.do_logistic_regression = do_logistic_regression
        self.generate_X(dim, size_dataset, power_cov, r_sigma, use_ortho_matrix)
        self.generate_Y()
        self.set_step_size()

    def regenerate_dataset(self):
        self.generate_X(self.dim, self.size_dataset, self.power_cov, self.r_sigma, self.use_ortho_matrix)
        self.generate_Y()

    def generate_X(self, dim: int, size_dataset: int, power_cov: int, r_sigma: int, use_ortho_matrix: bool):
        self.dim = dim
        self.power_cov = power_cov
        self.r_sigma = r_sigma
        self.use_ortho_matrix = use_ortho_matrix

        # Used to generate self.X
        self.upper_sigma = np.diag(np.array([1 / (i ** power_cov) for i in range(1, dim + 1)]), k=0)
        if self.use_ortho_matrix:
            self.ortho_matrix = ortho_group.rvs(dim=self.dim)

        self.size_dataset = size_dataset

        size_generator = max(self.size_dataset, MAX_SIZE_DATASET)
        self.X = multivariate_normal(np.zeros(self.dim), self.upper_sigma, size=size_generator)
        if use_ortho_matrix:
            self.X = self.X.dot(self.ortho_matrix.T)

        print("Memory footprint X", sys.getsizeof(self.X))
        print("Memory footprint SIGMA", sys.getsizeof(self.upper_sigma))

    def generate_Y(self):
        lower_sigma = 1  # Used only to introduce noise in the true labels.
        if self.r_sigma == 0:
            self.w_star = np.ones(self.dim)
        else:
            self.w_star = np.power(self.upper_sigma, self.r_sigma) @ np.ones(self.dim)

        if self.do_logistic_regression:
            self.Y = self.X @ self.w_star
            self.Y = np.random.binomial(1, expit(self.Y))
            self.Y[self.Y == 0] = -1
        else:
            size_generator = max(self.size_dataset, MAX_SIZE_DATASET)
            self.Y = self.X @ self.w_star + np.random.normal(0, lower_sigma, size=size_generator)


