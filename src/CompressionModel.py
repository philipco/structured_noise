"""
Created by Constantin Philippenko, 29th December 2021.

This python file provide facilities to quantize tensors.
"""
from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import bernoulli
from math import sqrt


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
    def __compress__(self, vector: np.ndarray, dim_to_use: int):
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

    def get_learning_rate(self, *args, **kwds):
        """Compute the learning rate.

        No argument if the operator already know the dimension.
        Set as unique argument a vector ig the dimension is unknow (in the case of DL)."""
        if self.level == 0:
            return 0
        if len(args) == 1:
            return self.constant / (self.__compute_omega_c__(args[0]) + 1)
        return self.constant / (self.omega_c + 1)


class TopKSparsification(CompressionModel):

    def __init__(self, level: int, dim: int = None, norm: int = 2, constant: int = 1):
        super().__init__(level, dim, norm, constant)
        if dim is not None:
            assert 0 <= level < 1, "k is a probability."
        self.biased = True

    def __compress__(self, vector: np.ndarray):
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


class RandomSparsification(CompressionModel):

    def __init__(self, level: int, dim: int = None, biased=False, norm: int = 2, constant: int = 1):
        """

        :param level: number of dimension to select at compression step
        :param dim: number of dimension in the dataset
        :param biased: set to True to used to biased version of this operators
        """
        self.biased = biased
        super().__init__(level, dim, norm, constant)
        assert 0 <= level <= 1, "The level must be expressed in percent."

    def __compress__(self, vector: np.ndarray):
        proba = self.level
        indices = bernoulli.rvs(proba, size=len(vector))
        compression = np.zeros_like(vector)
        for i in range(len(vector)):
            if indices[i]:
                compression[i] = vector[i] * [1 / proba, 1][self.biased]
        return compression

    def __omega_c_formula__(self, dim_to_use: int):
        proba = self.level
        # if self.biased:
        #     return 1 - proba
        return (1 - proba) / proba

    def get_name(self) -> str:
        if self.biased:
            return "RdkBsd"
        return "Rdk"


class SQuantization(CompressionModel):

# import numpy as np
# from CompressionModel import SQuantization, SQuantization2
# vec = np.array([-5,-15,11,9,8,0.01,0,-5,12,9,-7,3])
# compressor = SQuantization(1, len(vec))
# compressor2 = SQuantization2(1, len(vec))
# compressor.compress(vec)

    def __init__(self, level: int, dim: int = None, norm: int = 2, div_omega: int = 1, constant: int = 1):
        self.biased = False
        self.div_omega = div_omega
        super().__init__(level, dim, norm, constant)

    def __compress__(self, vector):

        # vector = torch.from_numpy(vector)
        # norm_x = torch.norm(vector, p=self.norm)
        # if norm_x == 0:
        #     return vector
        # all_levels = torch.floor(self.level * torch.abs(vector) / norm_x + torch.rand_like(vector)) / self.level
        # signed_level = torch.sign(vector) * all_levels
        # qtzt = signed_level * norm_x
        # return qtzt.numpy()

        norm_x = np.linalg.norm(vector, ord=self.norm)
        if norm_x == 0:
            return vector
        ratio = np.abs(vector) / norm_x
        p = ratio * self.level - np.floor(ratio * self.level)

        alea = np.random.binomial(1, p, len(vector))
        all_levels = (np.floor(self.level * ratio) + alea)/ self.level

        signed_level = np.sign(vector) * all_levels
        qtzt = signed_level * norm_x
        return qtzt

    def __omega_c_formula__(self, dim_to_use):
        return min(dim_to_use / (self.level * self.level * self.div_omega),
                   sqrt(dim_to_use) / (self.level * self.div_omega))

    def get_name(self) -> str:
        return "Qtzd"


class RandomDithering(CompressionModel):

    def __compress__(self, vector: np.ndarray) -> np.ndarray:

        xnorm = np.linalg.norm(vector)
        if xnorm == 0:
            return vector

        xx = np.random.uniform(0.0, 1.0, self.dim)
        xsign = np.sign(vector)
        x_int = np.floor(self.level * np.abs(vector) / xnorm + xx)
        x_cpmpress = xnorm / self.level * xsign * x_int
        return x_cpmpress

    def __omega_c_formula__(self, dim_to_use):
        return min(dim_to_use / (self.level * self.level),
                   sqrt(dim_to_use) / (self.level))

    def get_name(self) -> str:
        return "Ditering"


class SignSGD(CompressionModel):

    def __compress__(self, vector):
        # print(np.sign(vector))
        return np.sign(vector)

    def __omega_c_formula__(self, dim_to_use):
        return 2

    def get_name(self) -> str:
        return "Sign"
