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
        # TODO : with gaussian multiplication to check that the distribution is still ...
        X_compressed[i] = compressor.compress(X[i])
    cov_matrix = compute_empirical_covariance(X_compressed)
    return cov_matrix, X_compressed


def get_theoretical_cov(dataset: SyntheticDataset, compression_name: str):

    sigma = dataset.upper_sigma
    diag_sigma = np.diag(np.diag(sigma))

    if compression_name == "No compression":
        return sigma

    elif compression_name in ["Qtzd", "StabilizedQtz"]:
        return sigma - diag_sigma + np.sqrt(np.trace(sigma)) * np.sqrt(diag_sigma)

    elif compression_name == "Sparsification":
        ones = np.ones((dataset.dim, dataset.dim))
        P = dataset.LEVEL_RDK ** 2 * ones + (dataset.LEVEL_RDK - dataset.LEVEL_RDK ** 2) * np.eye(dataset.dim)
        return P * sigma / dataset.LEVEL_RDK ** 2

    elif compression_name == "Sketching":
        return sigma * (1 + 1 / dataset.sketcher.sub_dim) + np.trace(sigma) * np.identity(dataset.dim) / dataset.sketcher.sub_dim

    elif compression_name == "Rand1":
        return diag_sigma * dataset.dim

    elif compression_name == "PartialParticipation":
        return sigma / dataset.LEVEL_RDK

    elif compression_name == "DP":
        return sigma * (1 + np.trace(sigma) / dataset.LEVEL_RDK)

    # elif compression_name == "Ind DP":
    #     return sigma * 1 + np.trace(sigma) / dataset.LEVEL_RDK)

    return None


def compute_theoretical_trace(dataset: SyntheticDataset, compression_name: str):
    sigma = dataset.upper_sigma
    sigma_inv = np.linalg.inv(dataset.upper_sigma)

    if compression_name == "No compression":
        return dataset.dim

    elif compression_name == "Qtzd":
        residual1 = np.diag(np.diag(sigma)) @ sigma_inv
        residual2 = np.diag(np.sqrt(np.diag(sigma))) @ sigma_inv
        return dataset.dim + np.sqrt(np.trace(sigma)) * np.trace(residual2) - np.trace(residual1)

    elif compression_name == "Sparsification":
        residual1 = np.diag(np.diag(sigma)) @ sigma_inv
        p = dataset.sparsificator.sub_dim / dataset.dim
        return dataset.dim + (p - p ** 2) * np.trace(residual1) / (p ** 2)

    elif compression_name == "Sketching":
        trace_sketching = dataset.dim * (1 + 1 / dataset.sketcher.sub_dim)
        sub_sum = [np.sum([eig1 / eig2 for eig2 in dataset.eigenvalues]) for eig1 in dataset.eigenvalues]
        trace_sketching += np.sum(sub_sum) / dataset.sketcher.sub_dim
        return trace_sketching

    elif compression_name == "Rand1":
        return np.trace(dataset.dim * np.diag(np.diag(sigma)) @ sigma_inv)

    elif compression_name == "PartialParticipation":
        return dataset.dim / dataset.LEVEL_RDK

    return None