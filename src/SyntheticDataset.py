"""
Created by Constantin Philippenko, 10th January 2022.
"""
import copy
import random

import numpy as np
from numpy.random import multivariate_normal
from scipy.linalg import toeplitz
from scipy.special import expit
from scipy.stats import ortho_group

from src.CompressionModel import SQuantization, RandomSparsification, Sketching, find_level_of_quantization, \
    AllOrNothing, StabilizedQuantization, RandK, PoissonSparsification
from src.DataPreparationFromArtemis import generate_param
from src.JITProduct import diagonalization
from src.Utilities import print_mem_usage

MAX_SIZE_DATASET = 10**7

class AbstractDataset:

    def __init__(self, name: str = None) -> None:
        super().__init__()
        self.name = name

    def string_for_hash(self):
        hash = "N{0}-D{1}-P{2}-R{3}".format(self.size_dataset, self.dim, self.power_cov, self.r_sigma)
        if self.name:
            hash = "{0}-{1}".format(self.name, hash)
        if self.use_ortho_matrix:
            hash = "{0}-ortho".format(hash)
        return hash

    def define_compressors(self):

        self.LEVEL_QTZ = 1
        self.quantizator = SQuantization(self.LEVEL_QTZ, dim=self.dim)

        self.stabilized_quantizator = StabilizedQuantization(self.LEVEL_QTZ, dim=self.dim)

        self.LEVEL_RDK = self.quantizator.nb_bits_by_iter() / (32 * self.dim)
        self.sparsificator = RandomSparsification(self.LEVEL_RDK, dim=self.dim, biased=False)
        self.rand1 = RandK(1, dim=self.dim, biased=False)
        print("Level sparsification:", self.sparsificator.level)

        self.sketcher = Sketching(self.LEVEL_RDK, self.dim)
        self.rand_sketcher = Sketching(self.LEVEL_RDK, self.dim, randomized=True)

        self.all_or_nothinger = AllOrNothing(self.LEVEL_RDK, self.dim)

        print("No compr: {0:1.2f} bits/iter.".format(32 * self.dim))
        print("Quantiz: {0:1.2f} bits/iter.".format(self.quantizator.nb_bits_by_iter()))
        print("Sparsif: {0:1.2f} bits/iter.".format(self.sparsificator.nb_bits_by_iter()))
        print("Level qtz:", self.LEVEL_QTZ)
        print("Level rdk:", self.LEVEL_RDK)
        print("Subdimension cardinal:", self.sketcher.sub_dim)
        print("Qtz compression:", self.quantizator.omega_c)
        print("Rdk compression:", self.sparsificator.omega_c)

    def set_step_size(self):
        # We generate a dataset of a maximal size.
        size_generator = min(self.size_dataset, MAX_SIZE_DATASET)
        self.L = np.linalg.norm(self.X_complete.T @self.X_complete, ord=2) / self.size_dataset #np.max(self.eigenvalues)
        print("L=", self.L)

        R_SQUARE = np.trace(self.upper_sigma)
        print("4 * R_SQUARE", 4 * R_SQUARE)

        GAMMA_BACH_MOULINES = 1 / (4 * R_SQUARE)

        L_SPORTISSE = max([np.linalg.norm(self.X[k]) ** 2 for k in range(size_generator)])
        print("L SPORTISSE=", L_SPORTISSE)
        GAMMA_SPORTISSE = 1 / (2 * L_SPORTISSE)

        OPTIMAL_GAMMA_COMPR = 1 / (self.L * (1 + 2 * (self.quantizator.omega_c + 1)))
        print("Optimal gamma for compression:", OPTIMAL_GAMMA_COMPR)
        print("Gamma from Bach & Moulines, 13:", GAMMA_BACH_MOULINES)

        CONSTANT_GAMMA = .1 / (2 * self.L)
        print("Constant step size:", CONSTANT_GAMMA)

        print("Gamma sportisse:", GAMMA_SPORTISSE)

        self.gamma = OPTIMAL_GAMMA_COMPR

        print("Taken step size:", self.gamma)


class RealLifeDataset(AbstractDataset):

    def load_data(self, X, Y, do_logistic_regression: bool, name: str = None):
        self.do_logistic_regression = do_logistic_regression
        self.X, self.Y = X, Y
        self.upper_sigma = self.X.T @ self.X
        self.w_star = None
        self.size_dataset, self.dim = X.shape[0], X.shape[1]
        self.set_step_size()

BIAS = 2

class SyntheticDataset(AbstractDataset):

    def generate_dataset(self, dim: int, size_dataset: int, power_cov: int, r_sigma: int, use_ortho_matrix: bool,
                         do_logistic_regression: bool, eigenvalues: np.array = None):
        self.do_logistic_regression = do_logistic_regression
        self.generate_constants(dim, size_dataset, power_cov, r_sigma, use_ortho_matrix, eigenvalues=eigenvalues)
        self.define_compressors()
        self.generate_X()
        self.generate_Y()
        self.set_step_size()
        print_mem_usage("Just created the dataset ...")

    def generate_constants(self, dim: int, size_dataset: int, power_cov: int, r_sigma: int, use_ortho_matrix: bool,
                           eigenvalues: np.array = None):
        self.dim = dim
        self.power_cov = power_cov
        self.r_sigma = r_sigma
        self.use_ortho_matrix = use_ortho_matrix
        self.size_dataset = size_dataset

        # Construction of a covariance matrix
        self.upper_sigma = toeplitz(0.6 ** np.arange(0, self.dim))
        self.ortho_matrix = np.identity(self.dim)

        self.w0 = np.zeros(self.dim)
        self.w_star = generate_param(self.dim).numpy()

        # Used to generate self.X
        # if eigenvalues is None:
        #     self.eigenvalues = np.array([1 / (i ** self.power_cov) for i in range(1, self.dim + 1)])
        # else:
        #     self.eigenvalues = eigenvalues
        # self.upper_sigma = np.diag(self.eigenvalues, k=0)

        # np.random.seed(25)
        # if self.r_sigma == 0:
        #     self.w_star = np.random.normal(0, 1, size=self.dim) #np.ones(self.dim)
        # else:
        #     self.w_star = np.power(self.upper_sigma, self.r_sigma) @ np.ones(self.dim)

        if self.use_ortho_matrix:
            # theta = np.pi / 4
            # self.ortho_matrix = np.array([[np.cos(theta), - np.sin(theta)], [np.sin(theta), np.cos(theta)]]) #ortho_group.rvs(dim=self.dim)
            self.ortho_matrix = ortho_group.rvs(dim=self.dim)
            self.upper_sigma = self.ortho_matrix @ self.upper_sigma @ self.ortho_matrix.T
            self.Q, self.D = diagonalization(self.upper_sigma)
        else:
            self.ortho_matrix = np.identity(self.dim)

    def regenerate_dataset(self):
        self.generate_X()
        self.generate_Y()

    def generate_X(self):
        size_generator = min(self.size_dataset, MAX_SIZE_DATASET)
        self.X = multivariate_normal(np.zeros(self.dim), self.upper_sigma, size=size_generator)
        self.X_complete = copy.deepcopy(self.X)


    def generate_Y(self):
        lower_sigma = 0.4  # Used only to introduce noise in the true labels.
        size_generator = min(self.size_dataset, MAX_SIZE_DATASET)
        self.Y = self.X_complete @ self.w_star + np.random.normal(0, lower_sigma, size=size_generator)


