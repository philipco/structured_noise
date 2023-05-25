"""
Created by Constantin Philippenko, 17th January 2022.

Used to generate the figure in the paper which gives eigenvalues of the eigenvalues of the compressors's covariances.
"""
from typing import List

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm

from src.CompressionModel import Quantization, CompressionModel
from src.SyntheticDataset import SyntheticDataset, AbstractDataset
from src.TheoreticalCov import get_theoretical_cov
from src.federated_learning.Client import Client
from src.utilities.Utilities import create_folder_if_not_existing

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

SIZE_DATASET = 10**4
DIM = 100
POWER_COV = 4
R_SIGMA=0

NB_CLIENTS = 1

# In the case of heterogeneoux sigma and with non-diag H, check that random state of the orthogonal matrix is set to 5.
USE_ORTHO_MATRIX = False

HETEROGENEITY = "homog"

FONTSIZE = 20
LINESIZE = 4


def compute_diag(dataset: SyntheticDataset, compressor: CompressionModel) -> [np.ndarray, np.ndarray]:
    """Compute the diagonal of the given compressor's covariance."""

    X = dataset.X
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


def compute_diag_matrices(dataset: SyntheticDataset, clients: List[Client], dim: int) \
        -> [List[np.ndarray], AbstractDataset]:
    """Compute the diagonal of each compressor's covariance (considering the covariance's average of all clients)."""

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

    return all_diagonals, dataset


def compute_theoretical_diag(dataset: SyntheticDataset, nb_clients: int) -> List[np.ndarray]:
    """Compute the theoretical diagonal of each compressor's covariance (take as input the covariance's average of all
    clients)."""

    all_covariance = [get_theoretical_cov(dataset, nb_clients, "No compression"),
                      get_theoretical_cov(dataset, nb_clients, "Qtzd"),
                      get_theoretical_cov(dataset, nb_clients, "Sparsification"),
                      get_theoretical_cov(dataset, nb_clients, "Sketching"),
                      get_theoretical_cov(dataset, nb_clients, "Randh"),
                      get_theoretical_cov(dataset, nb_clients, "PartialParticipation")]

    if USE_ORTHO_MATRIX:
        for i in range(len(all_covariance)):
            all_covariance[i] = dataset.ortho_matrix.T.dot(all_covariance[i]).dot(dataset.ortho_matrix)

    all_diagonals = [np.diag(cov) for cov in all_covariance]
    return all_diagonals


if __name__ == '__main__':

    labels = ["no compr.", r"$1$-quantiz.", "sparsif.", "sketching", r"rand-$h$", "partial part."]

    clients = [Client(i, DIM, SIZE_DATASET // NB_CLIENTS, POWER_COV, NB_CLIENTS, USE_ORTHO_MATRIX, HETEROGENEITY)
               for i in range(NB_CLIENTS)]
    dataset = SyntheticDataset()
    all_diagonals, dataset = compute_diag_matrices(dataset, clients, dim=DIM, labels=labels)
    all_theoretical_diagonals = compute_theoretical_diag(dataset, len(clients))

    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    for (diagonal, label) in zip(all_diagonals, labels):
        axes[0].plot(np.log10(np.arange(1, DIM + 1)), np.log10(diagonal), label=label, lw = LINESIZE)
    for (diagonal, label) in zip(all_theoretical_diagonals, labels):
        axes[1].plot(np.log10(np.arange(1, DIM + 1)), np.log10(diagonal), label=label, lw = LINESIZE, linestyle="--")

    for ax in axes:
        ax.tick_params(axis='both', labelsize=FONTSIZE)
        ax.set_xlabel(r"$\log(i), \forall i \in \{1, ..., d\}$", fontsize=FONTSIZE)


    legend_line = [Line2D([0], [0], color="black", lw=LINESIZE, label='empirical'),
                   Line2D([0], [0], linestyle="--", color="black", lw=LINESIZE, label='theoretical')]

    l1 = axes[0].legend(loc='lower left', fontsize=FONTSIZE)
    l2 = axes[1].legend(handles=legend_line, loc="lower left", fontsize=FONTSIZE)
    axes[0].add_artist(l1)
    axes[1].add_artist(l2)


    axes[1].set_yticks([])
    axes[0].set_title('Empirical eigenvalues', fontsize=FONTSIZE)
    axes[1].set_title('Theoretical eigenvalues', fontsize=FONTSIZE)
    axes[0].set_ylabel(r"$\log(\mathrm{eig}(\mathfrak{C}(\mathcal{C}, p_M)_i)$", fontsize=FONTSIZE)
    folder = "../pictures/epsilon_eigenvalues"
    create_folder_if_not_existing(folder)

    hash = dataset.string_for_hash(nb_runs=1)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig("{0}/C{1}-{2}.pdf".format(folder, NB_CLIENTS, hash), bbox_inches='tight', dpi=600)

    plt.show()

