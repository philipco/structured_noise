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
from src.TheoreticalCov import get_theoretical_cov
from src.Utilities import create_folder_if_not_existing

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

SIZE_DATASET = 10**5
DIM = 100
POWER_COV = 4
R_SIGMA=0

USE_ORTHO_MATRIX = False
HETEROGENEITY = "homog"


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

    cov_matrix = dataset.ortho_matrix.T.dot(cov_matrix).dot(dataset.ortho_matrix)

    diag = np.diag(cov_matrix)
    return diag, cov_matrix


def compute_diag_matrices(dataset: SyntheticDataset, dim: int, labels):

    dataset.generate_constants(dim, size_dataset=SIZE_DATASET, power_cov=POWER_COV, r_sigma=R_SIGMA,
                               use_ortho_matrix=USE_ORTHO_MATRIX, heterogeneity=HETEROGENEITY)
    dataset.define_compressors()
    dataset.generate_X()

    no_compressor = SQuantization(0, dim=dim)

    my_compressors = [no_compressor, dataset.quantizator, dataset.sparsificator, dataset.rand_sketcher,
                      dataset.rand1, dataset.all_or_nothinger]

    all_diagonals = []
    for compressor in my_compressors:
        diag, cov_matrix = compute_diag(dataset, compressor)
        all_diagonals.append(diag)

    return all_diagonals, labels, dataset.string_for_hash()


def compute_theoretical_diag(dataset: SyntheticDataset, labels):

    ### No compression
    all_covariance = [get_theoretical_cov(dataset, "No compression"),
                      get_theoretical_cov(dataset, "Qtzd"),
                      get_theoretical_cov(dataset, "Sparsification"),
                      get_theoretical_cov(dataset, "Sketching"),
                      get_theoretical_cov(dataset, "Rand1"),
                      get_theoretical_cov(dataset, "AllOrNothing")]

    if USE_ORTHO_MATRIX:
        for i in range(len(all_covariance)):
            all_covariance[i] = dataset.ortho_matrix.T.dot(all_covariance[i]).dot(dataset.ortho_matrix)

    all_diagonals = [np.diag(cov) for cov in all_covariance]
    return all_diagonals, labels


if __name__ == '__main__':

    labels = ["no compr.", "quantiz.", "rdk", "gauss. proj.", "rand1", "all or noth."]

    dataset = SyntheticDataset()
    all_diagonals, labels, hash_dataset = compute_diag_matrices(dataset, dim=DIM, labels=labels)
    all_theoretical_diagonals, theoretical_labels = compute_theoretical_diag(dataset, labels=labels)

    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    for (diagonal, label) in zip(all_diagonals, labels):
        axes[0].plot(np.log10(np.arange(1, DIM + 1)), np.log10(diagonal), label=label, lw = 2)
    for (diagonal, label) in zip(all_theoretical_diagonals, theoretical_labels):
        axes[1].plot(np.log10(np.arange(1, DIM + 1)), np.log10(diagonal), label=label, lw = 2, linestyle="--")

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
    plt.savefig("{0}/{1}.eps".format(folder, hash_dataset), format='eps')

    plt.show()

