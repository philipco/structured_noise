"""
Created by Constantin Philippenko, 29th December 2021.

This python file provide facilities to quantize tensors.
"""
from abc import ABC, abstractmethod

import numpy as np
from math import sqrt

from scipy.stats import bernoulli, ortho_group, poisson


class CompressionModel(ABC):
    """
    The CompressionModel class declares the factory methods while subclasses provide the implementation of this methods.

    This class defines the operators of compression.
    """

    def __init__(self, level: int, dim: int = None, norm: int = 2, constant: int = 1):
        self.level = level
        self.dim = dim
        self.norm = norm
        self.constant = constant
        if dim is not None:
            self.omega_c = self.__compute_omega_c__(dim=dim)
        else:
            self.omega_c = None

    @abstractmethod
    def __compress__(self, vector: np.ndarray) -> np.ndarray:
        """Compresses a vector with the mechanism of the operator of compression."""
        pass

    def compress(self, vector: np.ndarray) -> np.ndarray:
        """Prepare a vector for compression, and compresses it.

        :param vector: The vector to be compressed.
        :return: The compressed vector
        """
        if self.level == 0:
            return vector
        vector, dim = vector, vector.shape[0]

        compressed_vector = self.__compress__(vector)
        return compressed_vector

    @abstractmethod
    def __omega_c_formula__(self, dim_to_use: int):
        """Proper implementation of the formula to compute omega_c.
        This formula is unique for each operator of compression."""
        pass

    def __compute_omega_c__(self, vector: np.ndarray = None, dim: int = None):
        """Compute the value of omega_c."""
        # If s==0, it means that there is no compression.
        # But for the need of experiments, we may need to compute the quantization constant associated with s=1.
        if self.level == 0:
            omega = 0
        else:
            omega = self.__omega_c_formula__(dim)
        return omega

    @abstractmethod
    def get_name(self) -> str:
        """Returns the name of the operator of compression."""
        pass

    @abstractmethod
    def nb_bits_by_iter(self):
        pass


class NoCompression(CompressionModel):

    def __compress__(self, vector: np.ndarray):
        return vector

    def __omega_c_formula__(self, dim_to_use: int):
        return 1

    def get_name(self) -> str:
        return "NoComp"

    def nb_bits_by_iter(self):
        return 32 * self.dim


class TopKSparsification(CompressionModel):

    def __init__(self, level: int, dim: int = None, norm: int = 2, constant: int = 1):
        super().__init__(level, dim, norm, constant)
        if dim is not None:
            assert 0 <= level < 1, "k is a probability."
        self.biased = True

    def __compress__(self, vector: np.ndarray) -> np.ndarray:
        indices = np.argpartition(vector, -self.level)[-self.level:]
        compression = np.zeros_like(vector)
        for i in indices:
            compression[i.item()] = vector[i]
        return compression

    def __omega_c_formula__(self, dim_to_use: int):
        proba = self.level
        return 1 - proba

    def get_name(self) -> str:
        return "Topk"

    def nb_bits_by_iter(self):
        return 32 * int(self.level * self.dim)


class RandomSparsification(CompressionModel):

    def __init__(self, level: int, dim: int = None, biased=False, norm: int = 2, constant: int = 1):
        """

        :param level: number of dimension to select at compression step
        :param dim: number of dimension in the dataset
        :param biased: set to True to used to biased version of this operators
        """
        self.biased = biased
        self.sub_dim = int(dim * level)
        super().__init__(level, dim, norm, constant)
        assert 0 <= level <= 1, "The level must be expressed in percent."

    def __compress__(self, vector: np.ndarray):
        proba = self.level
        indices = bernoulli.rvs(self.level, size=len(vector))
        compression = np.zeros_like(vector)
        for i in range(len(vector)):
            if indices[i]:
                compression[i] = vector[i] * [1 / proba, 1][self.biased]
        return compression

    def __omega_c_formula__(self, dim_to_use: int):
        proba = self.level
        if self.biased:
            return 1 - proba
        return (1 - proba) / proba

    def get_name(self) -> str:
        if self.biased:
            return "SparsificationBsd"
        return "Sparsification"

    def nb_bits_by_iter(self):
        return 32 * self.level * self.dim


class PoissonSparsification(CompressionModel):
    def __init__(self, level: int, dim: int = None, biased=False, norm: int = 2, constant: int = 1):
        """

        :param level: lambda param
        :param dim: number of dimension in the dataset
        :param biased: set to True to used to biased version of this operators
        """
        self.biased = biased
        print("Poisson level:", level)
        self.sub_dim = int(dim * level)
        super().__init__(level, dim, norm, constant)

    def __compress__(self, vector: np.ndarray):
        proba = self.level
        random_variable = poisson.rvs(self.level, size=len(vector))
        compression = np.zeros_like(vector)
        for i in range(len(vector)):
            # if indices[i]:
            compression[i] = random_variable[i] * vector[i] * [1 / proba, 1][self.biased]
        return compression

    def __omega_c_formula__(self, dim_to_use: int):
        proba = self.level
        if self.biased:
            return 1 - proba
        return (1 - proba) / proba

    def get_name(self) -> str:
        if self.biased:
            return "PoissonSparsif"
        return "PoissonSparsif"

    def nb_bits_by_iter(self):
        return 32 * (1 - np.e**-self.level) * self.dim


class RandK(RandomSparsification):

    def __init__(self, sub_dim: int, dim: int = None, biased=False, norm: int = 2, constant: int = 1):
        super().__init__(sub_dim / dim, dim, biased, norm, constant)
        self.sub_dim = sub_dim
        assert 1 <= sub_dim <= dim, "The sub dimension is not correct."

    def __compress__(self, vector: np.ndarray):
        indices = np.random.choice(self.dim, self.sub_dim)
        compression = np.zeros_like(vector)
        for i in range(len(indices)):
            compression[indices[i]] = vector[indices[i]] * [1 / self.level, 1][self.biased]
        return compression

    def get_name(self) -> str:
        if self.biased:
            return "Rand{0}-Bsd".format(self.sub_dim)
        return "Rand" + str(self.sub_dim)

    def nb_bits_by_iter(self):
        return 32 * self.level * self.dim


class Quantization(CompressionModel):

    def __init__(self, level: int, dim: int = None, norm: int = 2, div_omega: int = 1, constant: int = 1):
        self.biased = False
        self.div_omega = div_omega
        super().__init__(level, dim, norm, constant)

    def sample(self, p, n):
        return np.random.binomial(1, p, n)

    def __compress__(self, vector):

        norm_x = np.linalg.norm(vector, ord=self.norm)
        if norm_x == 0:
            return vector
        ratio = np.abs(vector) / norm_x
        p = ratio * self.level - np.floor(ratio * self.level)

        alea = self.sample(p, len(vector))
        all_levels = (np.floor(self.level * ratio) + alea)/ self.level

        signed_level = np.sign(vector) * all_levels
        qtzt = signed_level * norm_x
        return qtzt

    def __omega_c_formula__(self, dim_to_use):
        return min(dim_to_use / (self.level * self.level * self.div_omega),
                   sqrt(dim_to_use) / (self.level * self.div_omega))

    def get_name(self) -> str:
        return "Qtzd"

    def nb_bits_by_iter(self):
        if self.level == 0:
            return self.dim * 32
        frac = 2 * (self.level ** 2 + self.dim) / (self.level * (self.level+ np.sqrt(self.dim)))
        return (3 + 3 / 2) * np.log2(frac) * self.level * (self.level + np.sqrt(self.dim)) + 32


class CorrelatedQuantization(Quantization):

    def __init__(self, level: int, dim: int = None, norm: int = 2, constant: int = 1):
        self.biased = False
        super().__init__(level, dim, norm, constant)

    def __compress__(self, vector):
        norm_x = np.linalg.norm(vector, ord=self.norm)
        if norm_x == 0:
            return vector
        alea = np.random.uniform(0,1)
        all_levels = np.array([1 if alea < (np.abs(v) / norm_x) else 0 for v in vector])
        return all_levels * norm_x * np.sign(vector)

    def __omega_c_formula__(self, dim_to_use):
        return np.sqrt(dim_to_use)

    def get_name(self) -> str:
        return "Cor-Qtzd"


class AntiCorrelatedQuantization(Quantization):

    def __init__(self, level: int, dim: int = None, norm: int = 2, constant: int = 1):
        self.biased = False
        super().__init__(level, dim, norm, constant)

    def __compress__(self, vector):
        norm_x = np.linalg.norm(vector, ord=self.norm)
        if norm_x == 0:
            return vector
        alea = np.random.uniform(0,1, 1)
        alea = [alea if i % 2 == 0 else 1 - alea for i in range(len(vector))]
        scaled_vector = np.abs(vector) / norm_x
        all_levels = np.array([1 if alea[i] < scaled_vector[i] else 0 for i in range(len(vector))])
        return all_levels * norm_x * np.sign(vector)

    def __omega_c_formula__(self, dim_to_use):
        return np.sqrt(dim_to_use)

    def get_name(self) -> str:
        return "Anticor-Qtzd"


class PoissonQuantization(Quantization):

    def sample(self, p, n):
        return np.random.binomial(1, p, n)


class StabilizedQuantization(Quantization):

    def __compress__(self, vector):
        U = ortho_group.rvs(dim=self.dim)
        return U.T @ super().__compress__(U @ vector)

    def get_name(self) -> str:
        return "StabilizedQtz"


class Sketching(CompressionModel):

    def __init__(self, level: int, dim: int = None, randomized = False, type_proj = "gaussian", norm: int = 2, constant: int = 1):
        super().__init__(level, dim, norm, constant)
        self.biased = False
        self.sub_dim = int(dim * level)
        self.randomized = randomized
        self.type_proj = type_proj
        self.PHI = self.random_projector()
        self.PHI_INV = np.linalg.pinv(self.PHI)

    def random_projector(self):
        if self.type_proj == "gaussian":
            phi = np.random.normal(0, 1, size=(self.sub_dim, self.dim))
        elif self.type_proj == "sparse":
            phi = np.random.choice(np.array([-np.sqrt(3), 0, np.sqrt(3)]), p=[1/6, 2/3, 1/6], size=(self.sub_dim, self.dim))
        elif self.type_proj == "rdk":
            indices = np.random.choice(self.dim, self.sub_dim)
            phi = np.zeros((self.sub_dim, self.dim))
            for i in range(self.sub_dim):
                phi[i, indices[i]] = 1
            return phi
        else:
            raise ValueError("Type of projection is unknown.")
        return phi / np.sqrt(self.sub_dim)

    def __compress__(self, vector: np.ndarray) -> np.ndarray:
        if self.randomized:
            self.PHI = self.random_projector()
            self.PHI_INV = np.linalg.pinv(self.PHI)
        empirical_proba = self.sub_dim / self.dim
        return self.PHI_INV @ (self.PHI @ vector / empirical_proba)

    def __omega_c_formula__(self, dim_to_use: int):
        return (1 - self.level) / self.level

    def get_name(self) -> str:
        return "Sketching"

    def nb_bits_by_iter(self):
        return 32 * self.level * self.dim

class AllOrNothing(CompressionModel):

    def __init__(self, level: int, dim: int = None, norm: int = 2, constant: int = 1):
        """

        :param level: number of dimension to select at compression step
        :param dim: number of dimension in the dataset
        :param biased: set to True to used to biased version of this operators
        """
        super().__init__(level, dim, norm, constant)
        assert 0 <= level <= 1, "The level must be expressed in percent."

    def __compress__(self, vector: np.ndarray):
        b = np.random.binomial(1, self.level)
        return b * vector / self.level

    def __omega_c_formula__(self, dim_to_use: int):
        return (1 - self.level) / self.level

    def get_name(self) -> str:
        return "PartialParticipation"

    def nb_bits_by_iter(self):
        return 32 * self.level * self.dim


def find_level_of_quantization(dim, level_of_sparsification):
    """Given a level of sparsfication and a dimensionality, return the level of quantization to communicate a constant
    number of bits."""

    from scipy.optimize import fsolve

    def equation_to_solve(x):
        """x[0] = dim, x[1] = p"""
        frac = 2 * (x ** 2 + dim) / (x * (x + np.sqrt(dim)))
        return (3 + 3 / 2) * np.log(frac) * x * (x + np.sqrt(dim)) + 32 - 32 * level_of_sparsification * dim

    return np.round(fsolve(equation_to_solve, [2]))




