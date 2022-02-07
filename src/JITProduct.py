"""
Created by Constantin Philippenko, 7th February 2022.
"""

from numba import jit


@jit(nopython=True)
def wAw_product(alpha, w, A):
    return alpha * w.T @ A @ w


@jit(nopython=True)
def matrix_vector_product(A, x):
    return A @ x


@jit(nopython=True)
def scalar_product(x, y):
    return x @ y


@jit(nopython=True)
def vectorial_norm(x):
    return scalar_product(x, x)


@jit(nopython=True)
def constant_product(alpha, x):
    return alpha * x


@jit(nopython=True)
def minus(x, y):
    return x-y