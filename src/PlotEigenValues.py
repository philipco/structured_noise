"""
Created by Constantin Philippenko, 17th January 2022.
"""
import cmath

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.CompressionModel import SQuantization, RandomSparsification, Sketching
from src.SyntheticDataset import SyntheticDataset
from src.Utilities import create_folder_if_not_existing

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

SIZE_DATASET = 10**6
DIM = 100
POWER_COV = 4
R_SIGMA=0

USE_ORTHO_MATRIX = False


def prepare_sparsification(x, p):
    rademacher = np.random.binomial(1, 0.5, size=len(x))
    rademacher[rademacher == 0] = -1
    return x * (rademacher)


def compute_diag(dataset, compressor):

    X = dataset.X_complete

    X_compressed = X.copy()
    for i in tqdm(range(SIZE_DATASET)):
        X_compressed[i] = compressor.compress(X[i])

    cov_matrix = X_compressed.T.dot(X_compressed) / SIZE_DATASET

    if USE_ORTHO_MATRIX:
        cov_matrix = dataset.ortho_matrix.T.dot(cov_matrix).dot(dataset.ortho_matrix)

    diag = np.diag(cov_matrix)
    return diag, cov_matrix

def compute_diag_matrices(dataset: SyntheticDataset, dim: int):

    dataset.generate_constants(dim, size_dataset=SIZE_DATASET, power_cov=POWER_COV, r_sigma=R_SIGMA,
                       use_ortho_matrix=USE_ORTHO_MATRIX)
    dataset.define_compressors()
    dataset.generate_X()

    no_compressor = SQuantization(0, dim=dim)

    quantizator = SQuantization(1, dim=dim)

    p = 1 / (quantizator.omega_c + 1)
    print("Sparsification proba: ", p)
    sparsificator = RandomSparsification(p, dim, biased=False)

    gaussian_sketcher = Sketching(p, dim, type_proj = "gaussian")
    rand_gaussian_sketcher = Sketching(p, dim, randomized=True, type_proj = "gaussian")

    sparse_sketcher = Sketching(p, dim, type_proj="gaussian")
    rand_sparse_sketcher = Sketching(p, dim, randomized=True, type_proj="gaussian")

    my_compressors = [no_compressor, quantizator, sparsificator, rand_gaussian_sketcher]

    labels = ["no compr.", "quantiz.", "rdk", "gauss. proj."]#

    all_diagonals = []
    for compressor in my_compressors:
        diag, cov_matrix = compute_diag(dataset, compressor)
        all_diagonals.append(diag)

    return all_diagonals, labels, dataset.string_for_hash()


def compute_theoretical_diag(dataset: SyntheticDataset):
    labels = ["no comprs.", "quantiz.", "rdk", "gauss. proj."]

    ### No compression
    sigma = dataset.upper_sigma
    diag_sigma = np.diag(np.diag(sigma))
    all_covariance = [sigma]

    ### Quantization
    cov_qtz = sigma - diag_sigma + np.sqrt(np.trace(sigma)) * np.sqrt(diag_sigma)
    all_covariance.append(cov_qtz)

    ### Sparsification
    ones = np.ones((dataset.dim, dataset.dim))
    P = dataset.LEVEL_RDK **2 * ones + (dataset.LEVEL_RDK - dataset.LEVEL_RDK ** 2) * np.eye(dataset.dim)
    cov_rdk = P * sigma / dataset.LEVEL_RDK**2
    all_covariance.append(cov_rdk)

    ### Sketching
    cov_sketching = sigma * (1 + 1/dataset.sketcher.sub_dim) + np.trace(sigma) * np.identity(dataset.dim) / dataset.sketcher.sub_dim
    all_covariance.append(cov_sketching)

    if USE_ORTHO_MATRIX:
        for i in range(len(all_covariance)):
            all_covariance[i] = dataset.ortho_matrix.T.dot(all_covariance[i]).dot(dataset.ortho_matrix)

    all_diagonals = [np.diag(cov) for cov in all_covariance]
    return all_diagonals, labels


if __name__ == '__main__':

    dataset = SyntheticDataset()
    all_diagonals, labels, hash_dataset = compute_diag_matrices(dataset, dim=DIM)
    all_theoretical_diagonals, theoretical_labels = compute_theoretical_diag(dataset)

    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    for (diagonal, label) in zip(all_diagonals, labels):
        axes[0].plot(np.log10(np.arange(1, DIM + 1)), np.log10(diagonal), label=label, lw = 2)
    for (diagonal, label) in zip(all_theoretical_diagonals, theoretical_labels):
        axes[1].plot(np.log10(np.arange(1, DIM + 1)), np.log10(diagonal), label=label, lw = 2)

    for ax in axes:
        ax.tick_params(axis='both', labelsize=15)
        ax.legend(loc='best', fontsize=15)
        ax.set_xlabel(r"$\log(i), \forall i \in \{1, ..., d\}$", fontsize=15)
    axes[0].title.set_text('Empirical eigenvalues')
    axes[1].title.set_text('Theoretical eigenvalues')
    axes[0].set_ylabel(r"$\log(Diag(\frac{\mathcal C (X)^T.\mathcal C (X)}{n})_i)$", fontsize=15)
    plt.legend(loc='best', fontsize=15)
    folder = "pictures/epsilon_eigenvalues/"
    create_folder_if_not_existing(folder)
    # plt.savefig("{0}/{1}.eps".format(folder, hash_dataset), format='eps')

    plt.show()

