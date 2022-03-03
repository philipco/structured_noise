"""
Created by Constantin Philippenko, 17th January 2022.
"""

import numpy as np
import matplotlib.pyplot as plt

from src.CompressionModel import SQuantization, RandomSparsification
from src.SyntheticDataset import SyntheticDataset

SIZE_DATASET = 10**3
DIM = 100
POWER_COV = 2
R_SIGMA=0

USE_ORTHO_MATRIX = False


def compute_trace(dim: int) -> int:
    dataset = SyntheticDataset()
    dataset.generate_constants(dim, size_dataset=SIZE_DATASET, power_cov=POWER_COV, r_sigma=R_SIGMA,
                               use_ortho_matrix=USE_ORTHO_MATRIX)
    dataset.generate_X()
    X_quantized = dataset.X.copy()
    X_sparsed = dataset.X.copy()

    quantizator = SQuantization(1, dim=dim)

    p = 1 / (quantizator.omega_c + 1)
    print("Sparsification proba: ", p)
    sparsificator = RandomSparsification(p, dim, biased=True)

    for i in range(SIZE_DATASET):
        X_quantized[i] = quantizator.compress(dataset.X[i])
        X_sparsed[i] = sparsificator.compress(dataset.X[i])

    cov_matrix = dataset.X.T.dot(dataset.X) / SIZE_DATASET
    cov_matrix_qtz = X_quantized.T.dot(X_quantized) / SIZE_DATASET
    cov_matrix_sparse = X_sparsed.T.dot(X_sparsed) / SIZE_DATASET

    # if USE_ORTHO_MATRIX:
    #     cov_matrix = dataset.ortho_matrix.T.dot(cov_matrix).dot(dataset.ortho_matrix)
    #     cov_matrix_qtz = dataset.ortho_matrix.T.dot(cov_matrix_qtz).dot(dataset.ortho_matrix)
    #     cov_matrix_sparse = dataset.ortho_matrix.T.dot(cov_matrix_sparse).dot(dataset.ortho_matrix)

    trace = np.trace(cov_matrix.dot(np.linalg.inv(dataset.upper_sigma)))
    trace_qtz = np.trace(cov_matrix_qtz.dot(np.linalg.inv(dataset.upper_sigma)))
    trace_sparse = np.trace(cov_matrix_sparse.dot(np.linalg.inv(dataset.upper_sigma)))

    return trace, trace_qtz, trace_sparse


if __name__ == '__main__':

    print("Starting the script.")
    range_trace = np.arange(2, 100)

    all_trace, all_trace_qtz, all_trace_sparse = [], [], []

    for dim in range_trace:
        print()
        trace, trace_qtz, trace_sparse = compute_trace(dim)
        all_trace.append(trace)
        all_trace_qtz.append(trace_qtz)
        all_trace_sparse.append(trace_sparse)

    fig, ax = plt.subplots(figsize=(8, 7))
    plt.plot(range_trace, all_trace, label="No compression")
    plt.plot(range_trace, all_trace_qtz, label="Quantization")
    plt.plot(range_trace, all_trace_sparse, label="Sparsification")
    ax.tick_params(axis='both', labelsize=15)
    ax.legend(loc='best', fontsize=15)
    ax.set_xlabel(r"Dimension $d$", fontsize=15)
    ax.set_ylabel(r"$Trace(\frac{X^T.X \Sigma}{n})$", fontsize=15)

    print("Script completed.")

    plt.show()

