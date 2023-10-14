"""
Created by Constantin Philippenko, 17th January 2022.

Used to generate the figure in the paper which gives the trace of the compressors's covariances.
"""
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from src.CompressionModel import Quantization, CompressionModel
from src.SyntheticDataset import SyntheticDataset
from src.TheoreticalCov import compute_theoretical_trace
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
R_SIGMA=0

NB_CLIENTS = 1

POWER_COV = 4

START_DIM = 2
END_DIM = 100

FONTSIZE = 20
LINESIZE = 4

WITH_STANDARDISATION = True

COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:cyan"]

HETEROGENEITY = "homog" # "wstar" "sigma" "homog"


def compute_diag(dataset: SyntheticDataset, compressor: CompressionModel) -> np.ndarray:
    """Compute the diagonal of the given compressor's covariance."""

    X = dataset.X

    X_compressed = X.copy()
    for i in range(SIZE_DATASET):
        X_compressed[i] = compressor.compress(X[i])

    cov_matrix = X_compressed.T.dot(X_compressed) / SIZE_DATASET

    return cov_matrix


def compute_trace(clients: List[Client], dataset: SyntheticDataset, dim: int, use_ortho: bool, power_cov: int)\
        -> [List[float], SyntheticDataset]:
    """Compute the trace of all compressors' covariances."""

    upper_sigma = np.mean([clients[i].dataset.upper_sigma for i in range(len(clients))], axis=0)

    # Required in the case of covariance shift, we compute the mean of the covariance.
    dataset.generate_constants(dim, size_dataset=SIZE_DATASET, power_cov=power_cov, r_sigma=R_SIGMA,
                               use_ortho_matrix=use_ortho, heterogeneity=HETEROGENEITY, nb_clients=NB_CLIENTS,
                               client_id=0)
    dataset.define_compressors()
    dataset.power_cov = power_cov
    dataset.upper_sigma = upper_sigma
    dataset.generate_X()

    if WITH_STANDARDISATION:
        dataset.normalize()

    no_compressor = Quantization(0, dim=dim)

    my_compressors = [no_compressor, dataset.quantizator, dataset.sparsificator, dataset.sketcher,
                      dataset.rand1, dataset.all_or_nothinger]

    all_trace = []
    for compressor in my_compressors:
        cov_matrix = compute_diag(dataset, compressor)
        all_trace.append(np.trace(cov_matrix.dot(np.linalg.inv(dataset.second_moment_cov))))

    return all_trace, dataset


if __name__ == '__main__':

    labels = ["no compr.", r"$1$-quantiz.", "sparsif.", "sketching", r"rand-$h$", "partial part."]

    range_trace = np.arange(START_DIM, END_DIM)

    fig, axes = plt.subplots(1, 2, figsize=(10, 6))

    k = 0
    for use_ortho in [False, True]:
        trace_by_operators = [[] for i   in range(len(labels))]
        theoretical_trace_by_operators = [[] for i in range(len(labels))]

        for dim in range_trace:
            print(">>>>>>> Dimension:", dim)
            clients = [Client(i, dim, SIZE_DATASET // NB_CLIENTS, POWER_COV, NB_CLIENTS, use_ortho, HETEROGENEITY) for i in
                       range(NB_CLIENTS)]
            dataset = SyntheticDataset()
            all_trace, dataset = compute_trace(clients, dataset, dim, use_ortho, POWER_COV)
            for i in range(len(labels)):
                trace_by_operators[i].append(all_trace[i])
            all_theoretical_trace = [compute_theoretical_trace(dataset, "No compression"),
                                     compute_theoretical_trace(dataset, "Qtzd"),
                                     compute_theoretical_trace(dataset, "Sparsification"),
                                     compute_theoretical_trace(dataset, "Sketching"),
                                     compute_theoretical_trace(dataset, "Randh"),
                                     compute_theoretical_trace(dataset, "PartialParticipation")]
            for i in range(len(labels)):
                theoretical_trace_by_operators[i].append(all_theoretical_trace[i])

        for i in range(len(labels)):
            axes[k].plot(np.log10(range_trace), trace_by_operators[i], label=labels[i], lw=LINESIZE, color=COLORS[i])
            axes[k].plot(np.log10(range_trace), theoretical_trace_by_operators[i], label='_nolegend_', lw=LINESIZE,
                      color=COLORS[i], linestyle="--")

        axes[k].tick_params(axis='both', labelsize=FONTSIZE)
        axes[k].set_xlabel(r"$\log(i), \forall i \in \{1, ..., d\}$", fontsize=FONTSIZE)
        k += 1

    axes[0].set_ylabel(r"$\log(\mathrm{Tr}(\mathfrak{C}^{\mathrm{ania}} H^{-1}))$", fontsize=FONTSIZE)
    axes[0].legend(loc='best', fontsize=FONTSIZE)
    axes[1].set_yticks([])
    axes[0].set_title(r'$M$ diagonal', fontsize=FONTSIZE)
    axes[1].set_title(r'$M$ non-diagonal', fontsize=FONTSIZE)
    print("Script completed.")
    folder = "../pictures/trace/"
    create_folder_if_not_existing(folder)

    hash = dataset.string_for_hash(nb_runs=1)
    if WITH_STANDARDISATION:
        hash = "{0}-std".format(hash)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig("{0}/C{1}-{2}.pdf".format(folder, NB_CLIENTS, hash), bbox_inches='tight', dpi=600)


