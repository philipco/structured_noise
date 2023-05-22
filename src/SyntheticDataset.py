"""
Created by Constantin Philippenko, 10th January 2022.
"""
import copy

import numpy as np
from numpy.random import multivariate_normal
from scipy.stats import ortho_group, multivariate_t
from sklearn.preprocessing import StandardScaler

from src.CompressionModel import Quantization, RandomSparsification, Sketching, AllOrNothing, StabilizedQuantization, \
    RandK, CorrelatedQuantization, AntiCorrelatedQuantization, \
    DifferentialPrivacy, IndependantDifferentialPrivacy
from src.CustomDistribution import diamond_distribution
from src.JITProduct import diagonalization
from src.utilities.Utilities import print_mem_usage

MAX_SIZE_DATASET = 10**6

class AbstractDataset:

    def __init__(self, name: str = None) -> None:
        super().__init__()
        self.name = name
        self.real_dataset = False

    def string_for_hash(self, nb_runs: int, stochastic: bool = False, batch_size: int = 1, noiseless: bool = None,
                        reg: int = None, step: str = None):
        hash = "{4}runs-N{0}-D{1}-P{2}-{3}".format(self.size_dataset, self.dim, self.power_cov, self.heterogeneity, nb_runs)
        if self.name:
            hash = "{0}-{1}".format(self.name, hash)
        if self.use_ortho_matrix:
            hash = "{0}-ortho".format(hash)
        if not stochastic:
            hash = "{0}-full".format(hash)
        elif batch_size != 1:
            hash = "{0}-b{1}".format(hash, batch_size)
        if noiseless:
            hash = "{0}-noiseless".format(hash)
        if reg:
            hash = "{0}-reg{1}".format(hash, reg)
        if step:
            hash = "{0}-{1}".format(hash, step)
        return hash

    def define_compressors(self, s=None):

        # if omega is None:
        #     self.LEVEL_QTZ = 1
        # elif omega == 0:
        #     self.LEVEL_QTZ = 0
        # else:
        if s is None:
            self.LEVEL_QTZ = 1
        else:
            self.LEVEL_QTZ = s #round(max(math.sqrt(self.dim/omega**2), math.sqrt(self.dim) / omega))
        self.quantizator = Quantization(self.LEVEL_QTZ, dim=self.dim)

        self.correlated_quantizator = CorrelatedQuantization(level=self.LEVEL_QTZ, dim=self.dim)
        self.anti_cor_quantiz = AntiCorrelatedQuantization(level=self.LEVEL_QTZ, dim=self.dim)

        self.stabilized_quantizator = StabilizedQuantization(self.LEVEL_QTZ, dim=self.dim)

        self.LEVEL_RDK = 1/ (self.quantizator.omega_c + 1) #self.quantizator.nb_bits_by_iter() / (32 * self.dim)
        self.sparsificator = RandomSparsification(self.LEVEL_RDK, dim=self.dim, biased=False)
        self.rand1 = RandK(int(self.dim * self.LEVEL_RDK) if self.dim * self.LEVEL_RDK > 1 else 1, dim=self.dim, biased=False)

        self.dp = DifferentialPrivacy(self.LEVEL_RDK, dim=self.dim)
        self.ind_dp = IndependantDifferentialPrivacy(self.LEVEL_RDK, dim=self.dim)

        print("Level sparsification:", self.sparsificator.level)

        self.sketcher = Sketching(self.LEVEL_RDK, self.dim, randomized=True)

        self.all_or_nothinger = AllOrNothing(self.LEVEL_RDK, self.dim)

        print("No compr: {0:1.2f} bits/iter.".format(32 * self.dim))
        print("Quantiz: {0:1.2f} bits/iter.".format(self.quantizator.nb_bits_by_iter()))
        print("Sparsif: {0:1.2f} bits/iter.".format(self.sparsificator.nb_bits_by_iter()))
        print("Level qtz:", self.LEVEL_QTZ)
        print("Level rdk:", self.LEVEL_RDK)
        print("Subdimension cardinal:", self.sketcher.sub_dim)
        print("Omega quantization:", self.quantizator.omega_c)
        print("Omega sparsification:", self.sparsificator.omega_c)
        print("Omega sketching:", self.sketcher.omega_c)
        print("Omega rand1:", self.rand1.omega_c)

    def set_step_size(self):
        # We generate a dataset of a maximal size.
        self.L = np.max(self.eigenvalues)
        print("L=", self.L)

        self.trace = np.trace(self.upper_sigma)
        print("4 * r_square", 4 * self.trace)

        GAMMA_BACH_MOULINES = 1 / (4 * self.trace)

        OPTIMAL_GAMMA_COMPR = 1 / (self.L * (1 + 2 * (self.sparsificator.omega_c + 1)))
        print("Optimal gamma for compression:", OPTIMAL_GAMMA_COMPR)
        print("Gamma from Bach & Moulines, 13:", GAMMA_BACH_MOULINES)

        CONSTANT_GAMMA = .1 / (2 * self.L)
        print("Constant step size:", CONSTANT_GAMMA)

        self.gamma = 1 / ((self.sparsificator.omega_c + 1) * self.trace)

        print("Taken step size:", self.gamma)

class SyntheticDataset(AbstractDataset):

    def generate_dataset(self, dim: int, size_dataset: int, power_cov: int, r_sigma: int, nb_clients: int,
                         use_ortho_matrix: bool, do_logistic_regression: bool, heterogeneity: str,
                         client_id: int, eigenvalues: np.array = None, w0_seed: int = 42, lower_sigma: int = None):
        self.do_logistic_regression = do_logistic_regression
        self.generate_constants(dim, size_dataset, power_cov, r_sigma, nb_clients, use_ortho_matrix,
                                client_id=client_id, eigenvalues=eigenvalues, heterogeneity=heterogeneity,
                                w0_seed=w0_seed, lower_sigma=lower_sigma)
        self.define_compressors()
        self.generate_X()
        self.generate_Y()
        self.set_step_size()
        print_mem_usage("Just created the dataset ...")

    def generate_constants(self, dim: int, size_dataset: int, power_cov: int, r_sigma: int, nb_clients: int,
                           use_ortho_matrix: bool, heterogeneity: str, client_id: int, eigenvalues: np.array = None,
                           w0_seed: int = 42, lower_sigma: int = None):
        self.dim = dim
        self.nb_clients = nb_clients
        if lower_sigma is None:
            self.lower_sigma = self.nb_clients  # Used only to introduce noise in the true labels.
        else:
            self.lower_sigma = lower_sigma
        if heterogeneity == "sigma":
            self.power_cov = np.random.choice([3,4,5,6]) # for sigma
        else:
            self.power_cov = power_cov
        self.r_sigma = r_sigma
        self.use_ortho_matrix = use_ortho_matrix
        self.size_dataset = size_dataset
        self.heterogeneity = heterogeneity

        if w0_seed is not None:
            self.w0 = np.zeros(self.dim)
        else:
            self.w0 = multivariate_normal(np.zeros(self.dim), np.identity(self.dim) /self.dim)

        if self.heterogeneity == "wstar":
            self.w_star = np.random.normal(0, 10, size=self.dim)
        else:
            self.w_star = np.ones(self.dim)

        # Used to generate self.X
        if eigenvalues is None:
            if self.power_cov == "isotropic":
                self.eigenvalues = np.ones(self.dim)
            elif self.power_cov == "gap":
                self.eigenvalues = np.array([1 if i < self.dim // 2 else 1 / i for i in range(self.dim)])
            else:
                self.eigenvalues = np.array([1 / (i ** self.power_cov) for i in range(1, self.dim + 1)])
        else:
            self.eigenvalues = eigenvalues

        self.upper_sigma = np.diag(self.eigenvalues, k=0) #toeplitz(0.6 ** np.arange(0, self.dim)) #

        if self.use_ortho_matrix:
            # We fix the rotation matrix only when in dimension (for sake of TCL's plots clarity).
            if self.dim == 2:
                theta = np.pi / 8
                self.ortho_matrix = np.array([[np.cos(theta), - np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            else:
                if self.heterogeneity == "sigma":
                    self.ortho_matrix = ortho_group.rvs(dim=self.dim)
                else:
                    # Warning : if I print the eigenvalues, I need to have the same orthogonal matrix for all clients!
                    self.ortho_matrix = ortho_group.rvs(dim=self.dim, random_state=5)
            self.upper_sigma = self.ortho_matrix @ self.upper_sigma @ self.ortho_matrix.T
            self.Q, self.D = diagonalization(self.upper_sigma)
        else:
            self.ortho_matrix = np.identity(self.dim)

        self.center = np.zeros(self.dim)  # self.eigenvalues if self.power_cov > 0 else self.eigenvalues * 0.05
        # self.center[1] = 10
        m_carre = np.kron(self.center, self.center).reshape((self.dim, self.dim))
        self.second_moment_cov = self.upper_sigma + m_carre

    def regenerate_dataset(self):
        self.generate_X()
        self.generate_Y()

    def generate_X(self, features_distribution: str = "normal"):
        size_generator = min(self.size_dataset, MAX_SIZE_DATASET)
        if features_distribution == "diamond":
            self.X = diamond_distribution(size_generator)
            self.upper_sigma = np.identity(2) / 2
        elif features_distribution == "cauchy":
            self.X = multivariate_t.rvs(np.zeros(self.dim), self.upper_sigma, size=size_generator, df=2)
        elif features_distribution == "normal":
            self.X = multivariate_normal(self.center, self.upper_sigma, size=size_generator)
        else:
            raise ValueError("Unknow features distribution.")
        self.X_complete = copy.deepcopy(self.X)
        self.Xcarre = self.X_complete.T @ self.X_complete / size_generator

    def generate_Y(self):
        size_generator = min(self.size_dataset, MAX_SIZE_DATASET)
        self.Y = self.X_complete @ self.w_star
        self.epsilon = np.random.normal(0, np.sqrt(self.lower_sigma), size=size_generator)
        self.Y += self.epsilon
        self.Z = self.X_complete.T @ self.Y / size_generator

    def normalize(self):
        standardize_data = StandardScaler().fit_transform(self.X_complete)
        self.X_complete = standardize_data

