"""
Created by Constantin Philippenko, 17th January 2022.
"""
from typing import List

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.CompressionModel import Quantization
from src.SyntheticDataset import SyntheticDataset
from src.TheoreticalCov import get_theoretical_cov
from src.utilities.Utilities import create_folder_if_not_existing
from src.federated_learning.Client import Client

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

SIZE_DATASET = 10**4
DIM = 100
POWER_COV = 1
R_SIGMA=0

NB_CLIENTS = 1
EIGENVALUES = None #np.array([1,0.001]) #np.array([0.1,0.1, 0.00001,0.00001,0.00001,0.00001,0.00001, 0.00001, 0.00001])

# In the case of heterogeneoux sigma and with non-diag H, check that random state of the orthogonal matrix is set to 5.
USE_ORTHO_MATRIX = True
HETEROGENEITY = "homog"

FONTSIZE = 20
LINESIZE = 4


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

    # We diagonalize the covariance matrix by using the rotation matrix Q that was used to generate the dataset.
    # We don't compute the eigenvalues using np.linalg.eig as it introduces some errors.
    cov_matrix = dataset.ortho_matrix.T.dot(cov_matrix).dot(dataset.ortho_matrix)

    eigvalues, _ = np.linalg.eig(cov_matrix)
    diag = np.diag(cov_matrix)
    return diag, cov_matrix # Warning: if the eigenvalues are increasing, don't forget to sort the diag.


def compute_diag_matrices(dataset: SyntheticDataset, clients: List[Client], dim: int, labels):

    upper_sigma = np.mean([clients[i].dataset.upper_sigma for i in range(len(clients))], axis=0)

    dataset.generate_constants(dim, size_dataset=SIZE_DATASET, power_cov=POWER_COV, r_sigma=R_SIGMA,
                               use_ortho_matrix=USE_ORTHO_MATRIX, heterogeneity=HETEROGENEITY, client_id=0,
                               nb_clients=NB_CLIENTS)
    dataset.define_compressors()
    dataset.power_cov = POWER_COV
    dataset.upper_sigma = upper_sigma
    dataset.generate_X()

    no_compressor = Quantization(0, dim=dim)

    my_compressors = [no_compressor, dataset.quantizator, dataset.sparsificator, dataset.sketcher, dataset.rand1,
                      dataset.all_or_nothinger]

    all_diagonals = []
    for compressor in my_compressors:
        diag, cov_matrix = compute_diag(dataset, compressor)
        all_diagonals.append(diag)

    return all_diagonals, labels, dataset


def compute_theoretical_diag(dataset: SyntheticDataset, nb_clients, labels):

    ### No compression
    all_covariance = [get_theoretical_cov(dataset, nb_clients, "No compression"),
                      get_theoretical_cov(dataset, nb_clients, "Qtzd"),
                      get_theoretical_cov(dataset, nb_clients, "Sparsification"),
                      get_theoretical_cov(dataset, nb_clients, "Sketching"),
                      get_theoretical_cov(dataset, nb_clients, "Rand1"),
                      get_theoretical_cov(dataset, nb_clients, "PartialParticipation")]

    if USE_ORTHO_MATRIX:
        for i in range(len(all_covariance)):
            all_covariance[i] = dataset.ortho_matrix.T.dot(all_covariance[i]).dot(dataset.ortho_matrix)

    all_diagonals = [np.diag(cov) for cov in all_covariance]
    return all_diagonals, labels


if __name__ == '__main__':

    labels = ["no compr.", "1-quantiz.", "sparsif.", "sketching", "rand-1", "partial part."]

    clients = [Client(i, DIM, SIZE_DATASET // NB_CLIENTS, POWER_COV, NB_CLIENTS, USE_ORTHO_MATRIX, HETEROGENEITY,
                      eigenvalues=EIGENVALUES)
               for i in range(NB_CLIENTS)]
    dataset = SyntheticDataset()
    all_diagonals, labels, dataset = compute_diag_matrices(dataset, clients, dim=DIM, labels=labels)
    all_theoretical_diagonals, theoretical_labels = compute_theoretical_diag(dataset, len(clients), labels=labels)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for (diagonal, label) in zip(all_diagonals, labels):
        axes[0].plot(np.log10(np.arange(1, DIM + 1)), np.log10(diagonal), label=label, lw = LINESIZE)
    for (diagonal, label) in zip(all_theoretical_diagonals, theoretical_labels):
        axes[1].plot(np.log10(np.arange(1, DIM + 1)), np.log10(diagonal), label=label, lw = LINESIZE, linestyle="--")

    for ax in axes:
        ax.tick_params(axis='both', labelsize=FONTSIZE)
        ax.legend(loc='lower left', fontsize=FONTSIZE)
        ax.set_xlabel(r"$\log(i), \forall i \in \{1, ..., d\}$", fontsize=FONTSIZE)
    axes[0].set_title('Empirical eigenvalues', fontsize=FONTSIZE)
    axes[1].set_title('Theoretical eigenvalues', fontsize=FONTSIZE)
    axes[0].set_ylabel(r"$\log(\mathrm{eig}(\mathfrak{C}^{\mathrm{ania}})_i)$", fontsize=FONTSIZE)
    plt.legend(loc='lower left', fontsize=FONTSIZE)
    folder = "pictures/epsilon_eigenvalues/"
    create_folder_if_not_existing(folder)

    hash = dataset.string_for_hash(nb_runs=1)
    plt.savefig("{0}/C{1}-{2}.pdf".format(folder, NB_CLIENTS, hash), bbox_inches='tight', dpi=600)

    plt.show()

