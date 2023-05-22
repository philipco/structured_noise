"""Created by Constantin Philippenko, 15th April 2022."""

import numpy as np

from src.SyntheticDataset import SyntheticDataset


def compute_inversion(matrix):
    return np.linalg.inv(matrix)


def compute_limit_distrib(inv_sigma, error_cov):
    return inv_sigma @ error_cov @ inv_sigma


def compute_empirical_covariance(random_vector):
    return random_vector.T.dot(random_vector) / random_vector.shape[0]


def compress_and_compute_covariance(dataset, compressor):
    X = dataset.X_complete
    X_compressed = X.copy()
    for i in range(len(X)):
        X_compressed[i] = compressor.compress(X[i])
    cov_matrix = compute_empirical_covariance(X_compressed)
    return cov_matrix, X_compressed


def get_theoretical_cov(dataset: SyntheticDataset, nb_clients, compression_name: str):

    sigma = dataset.second_moment_cov / nb_clients
    diag_sigma = np.diag(np.diag(sigma))

    if compression_name == "No compression":
        return sigma

    elif compression_name in ["Qtzd", "StabilizedQtz"]:
        return sigma - diag_sigma + np.sqrt(np.trace(sigma)) * np.sqrt(diag_sigma)

    elif compression_name == "Sparsification":
        return sigma + diag_sigma * (1 - dataset.LEVEL_RDK) / dataset.LEVEL_RDK

    elif compression_name == "Sketching":
        alpha = dataset.sketcher.sub_dim * (dataset.sketcher.sub_dim + 2) / ( dataset.dim * (dataset.dim + 2))
        beta = (dataset.dim - dataset.sketcher.sub_dim) * dataset.sketcher.sub_dim / (dataset.dim ** 2 * (dataset.dim - 1))
        formula = (sigma * (alpha - beta)  + np.trace(sigma) * np.identity(dataset.dim) * beta)
        print(dataset.sketcher.sub_dim**2)
        return dataset.dim**2 * formula / dataset.sketcher.sub_dim**2

    elif compression_name == "Randh":
        sub_dim = dataset.rand1.sub_dim
        p1 = sub_dim / dataset.dim
        p2 = (sub_dim - 1) / (dataset.dim - 1)
        return (p2 * sigma + (1 - p2) * diag_sigma) / p1

    elif compression_name == "PartialParticipation":
        return sigma / dataset.LEVEL_RDK

    return None


def compute_theoretical_trace(dataset: SyntheticDataset, compression_name: str):
    sigma = dataset.second_moment_cov
    diag_sigma = np.diag(np.diag(sigma))
    sigma_inv = np.linalg.inv(sigma)

    if compression_name == "No compression":
        return dataset.dim

    elif compression_name == "Qtzd":
        residual1 = diag_sigma @ sigma_inv
        residual2 = np.diag(np.sqrt(np.diag(sigma))) @ sigma_inv
        return dataset.dim + np.sqrt(np.trace(sigma)) * np.trace(residual2) - np.trace(residual1)

    elif compression_name == "Sparsification":
        residual1 = diag_sigma @ sigma_inv
        p = dataset.sparsificator.level
        return dataset.dim + (p - p ** 2) * np.trace(residual1) / (p ** 2)

    elif compression_name == "Sketching":
        alpha = dataset.sketcher.sub_dim * (dataset.sketcher.sub_dim + 2) / (dataset.dim * (dataset.dim + 2))
        beta = (dataset.dim - dataset.sketcher.sub_dim) * dataset.sketcher.sub_dim / (dataset.dim ** 2 * (dataset.dim - 1))
        formula = (sigma * (alpha - beta) + np.trace(sigma) * np.identity(dataset.dim) * beta)
        print(dataset.sketcher.sub_dim ** 2)
        return np.trace((dataset.dim ** 2 * formula / dataset.sketcher.sub_dim ** 2) @ sigma_inv)

    elif compression_name == "Randh":
        sub_dim = dataset.rand1.sub_dim
        p1 = sub_dim / dataset.dim
        p2 = (sub_dim -1) / (dataset.dim - 1)
        return np.trace( ((p2 * sigma + (1 - p2) * diag_sigma) / p1) @ sigma_inv)

    elif compression_name == "PartialParticipation":
        return dataset.dim / dataset.LEVEL_RDK

    return None