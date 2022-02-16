"""
Created by Constantin Philippenko, 17th January 2022.
"""

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.CompressionModel import SQuantization, RandomSparsification
from src.SyntheticDataset import SyntheticDataset

SIZE_DATASET = 10**5
DIM = 1000
POWER_COV = 4
R_SIGMA=0

USE_ORTHO_MATRIX = True


def compute_diag_matrices(dim: int):

    dataset = SyntheticDataset()
    dataset.generate_constants(dim, size_dataset=SIZE_DATASET, power_cov=POWER_COV, r_sigma=R_SIGMA,
                       use_ortho_matrix=USE_ORTHO_MATRIX)
    dataset.generate_X()
    X_quantized = dataset.X.copy()
    X_sparsed = dataset.X.copy()

    quantizator = SQuantization(1, dim=dim)

    p = 1 / (quantizator.omega_c + 1)
    print("Sparsification proba: ", p)
    sparsificator = RandomSparsification(p, dim, biased=False)

    for i in tqdm(range(SIZE_DATASET)):
        X_quantized[i] = quantizator.compress(dataset.X[i])
        X_sparsed[i] = sparsificator.compress(dataset.X[i])

    cov_matrix = dataset.X.T.dot(dataset.X) / SIZE_DATASET
    cov_matrix_qtz = X_quantized.T.dot(X_quantized) / SIZE_DATASET
    cov_matrix_sparse = X_sparsed.T.dot(X_sparsed) / SIZE_DATASET

    if USE_ORTHO_MATRIX:
        cov_matrix = dataset.ortho_matrix.T.dot(cov_matrix).dot(dataset.ortho_matrix)
        cov_matrix_qtz = dataset.ortho_matrix.T.dot(cov_matrix_qtz).dot(dataset.ortho_matrix)
        cov_matrix_sparse = dataset.ortho_matrix.T.dot(cov_matrix_sparse).dot(dataset.ortho_matrix)

    # diag, _ = np.linalg.eig(cov_matrix)
    # diag_qtz, _ = np.linalg.eig(cov_matrix_qtz)
    # # diag_sparse, _ = np.linalg.eig(cov_matrix_sparse)
    # diag = np.diag(diag)
    # diag_qtz = np.diag(diag_qtz)
    # diag_sparse = np.diag(diag_sparse)

    diag = np.diag(cov_matrix)
    diag_qtz = np.diag(cov_matrix_qtz)
    diag_sparse = np.abs(np.diag(cov_matrix_sparse))

    return cov_matrix, cov_matrix_qtz, cov_matrix_sparse, diag, diag_qtz, diag_sparse


if __name__ == '__main__':

    cov_matrix, cov_matrix_qtz, cov_matrix_sparse, diag, diag_qtz, diag_sparse = compute_diag_matrices(dim=DIM)

    plt.imshow(cov_matrix)
    plt.colorbar()
    plt.title("No compression", fontsize=15)
    plt.show()

    plt.imshow(cov_matrix_qtz)
    plt.colorbar()
    plt.title("Quantization", fontsize=15)
    plt.show()

    plt.imshow(cov_matrix_sparse)
    plt.colorbar()
    plt.title("Sparsification", fontsize=15)
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 7))
    plt.plot(np.log10(np.arange(1, DIM + 1)), np.log10(diag), label="No compression")
    plt.plot(np.log10(np.arange(1, DIM + 1)), np.log10(diag_qtz), label="Quantization")
    plt.plot(np.log10(np.arange(1, DIM + 1)), np.log10(diag_sparse), label="Sparsification")
    ax.tick_params(axis='both', labelsize=15)
    ax.legend(loc='best', fontsize=15)
    ax.set_xlabel(r"$\log(i), \forall i \in \{1, ..., d\}$", fontsize=15)
    ax.set_ylabel(r"$\log(Diag(\frac{X^T.X}{n})_i)$", fontsize=15)
    plt.legend(loc='best', fontsize=15)

    plt.show()

