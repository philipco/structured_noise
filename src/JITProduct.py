"""
Created by Constantin Philippenko, 7th February 2022.
"""
import numpy as np
from numba import jit


@jit(nopython=True)
def wAw_product(alpha: int, w: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Compute alpha * w.T @ A @ w."""
    return alpha * w.T @ A @ w


@jit(nopython=True)
def matrix_vector_product(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Compute A @ x."""
    return A @ x


@jit(nopython=True)
def scalar_product(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute x @ y."""
    return x @ y


@jit(nopython=True)
def vectorial_norm(x: np.ndarray) -> np.ndarray:
    """Compute norm of x."""
    return scalar_product(x, x)


@jit(nopython=True)
def constant_product(alpha: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Compute alpha * x."""
    return alpha * x


@jit(nopython=True)
def minus(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute x - y."""
    return x - y


def diagonalization(matrix: np.ndarray) -> [np.ndarray, np.ndarray]:
    """Diagonalize a matrix."""
    D, Q = np.linalg.eig(matrix)
    return Q.real, np.diag(D).real