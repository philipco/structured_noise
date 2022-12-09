"""
Created by Constantin Philippenko, 17th January 2022.
"""
from typing import List

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from src.CompressionModel import Quantization, RandomSparsification, Sketching
from src.SyntheticDataset import SyntheticDataset
from src.TheoreticalCov import compute_theoretical_trace
from src.Utilities import create_folder_if_not_existing
from src.federated_learning.Client import Client

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

SIZE_DATASET = 10**4
DIM = 200
POWER_COV = 4
R_SIGMA=0

NB_CLIENTS = 1

START_DIM = 2
END_DIM = 100

FONTSIZE = 17
LINESIZE = 3

COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:brown", "tab:purple", "tab:cyan"]

USE_ORTHO_MATRIX = True
HETEROGENEITY = "homog" # "wstar" "sigma" "homog"


def compute_diag(dataset, compressor):

    X = dataset.X_complete

    X_compressed = X.copy()
    for i in range(SIZE_DATASET):
        X_compressed[i] = compressor.compress(X[i])

    cov_matrix = X_compressed.T.dot(X_compressed) / SIZE_DATASET

    return cov_matrix


def compute_trace(dataset: SyntheticDataset, dim: int) -> [List[float], SyntheticDataset]:

    upper_sigma = np.mean([clients[i].dataset.upper_sigma for i in range(len(clients))], axis=0)

    dataset.generate_constants(dim, size_dataset=SIZE_DATASET, power_cov=POWER_COV, r_sigma=R_SIGMA,
                               use_ortho_matrix=USE_ORTHO_MATRIX, heterogeneity=HETEROGENEITY, nb_clients=NB_CLIENTS)
    dataset.define_compressors()
    dataset.power_cov = POWER_COV
    dataset.upper_sigma = upper_sigma
    dataset.generate_X()

    no_compressor = Quantization(0, dim=dim)

    my_compressors = [no_compressor, dataset.quantizator, dataset.sparsificator, dataset.sketcher,
                      dataset.rand1, dataset.all_or_nothinger]

    all_trace = []
    for compressor in my_compressors:
        cov_matrix = compute_diag(dataset, compressor)
        all_trace.append(np.trace(cov_matrix.dot(np.linalg.inv(dataset.upper_sigma))))

    return all_trace, dataset


if __name__ == '__main__':

    print("Starting the script.")

    labels = ["no compr.", "1-quantiz.", "sparsif.", "sketching", "rand-1", "partial part."]

    range_trace = np.arange(START_DIM, END_DIM)

    trace_by_operators = [[] for i in range(len(labels))]
    theoretical_trace_by_operators = [[] for i in range(len(labels))]

    for dim in range_trace:
        clients = [Client(dim, SIZE_DATASET // NB_CLIENTS, POWER_COV, NB_CLIENTS, USE_ORTHO_MATRIX, HETEROGENEITY) for i in
                   range(NB_CLIENTS)]
        dataset = SyntheticDataset()
        all_trace, dataset = compute_trace(dataset, dim)
        for i in range(len(labels)):
            trace_by_operators[i].append(all_trace[i])
        all_theoretical_trace = [compute_theoretical_trace(dataset, "No compression"),
                                 compute_theoretical_trace(dataset, "Qtzd"),
                                 compute_theoretical_trace(dataset, "Sparsification"),
                                 compute_theoretical_trace(dataset, "Sketching"),
                                 compute_theoretical_trace(dataset, "Rand1"),
                                 compute_theoretical_trace(dataset, "PartialParticipation")]
        for i in range(len(labels)):
            theoretical_trace_by_operators[i].append(all_theoretical_trace[i])


    fig, axes = plt.subplots(figsize=(6, 6))
    for i in range(len(labels)):
        axes.plot(np.log10(range_trace), np.log10(trace_by_operators[i]), label=labels[i], lw=LINESIZE, color=COLORS[i])
        axes.plot(np.log10(range_trace), np.log10(theoretical_trace_by_operators[i]), label='_nolegend_', lw=LINESIZE,
                  color=COLORS[i], linestyle="--")

    axes.tick_params(axis='both', labelsize=FONTSIZE)
    axes.legend(loc='best', fontsize=FONTSIZE)
    axes.set_xlabel(r"$\log(i), \forall i \in \{1, ..., d\}$", fontsize=FONTSIZE)
    # axes.set_title('Empirical (plain) vs theoretical trace (dashed)')
    axes.set_ylabel(r"$\log(\mathrm{Tr}(\mathfrak{C}_{\mathrm{emp.}} H^{-1})_i)$", fontsize=FONTSIZE)

    print("Script completed.")
    folder = "pictures/trace/"
    create_folder_if_not_existing(folder)

    hash = dataset.string_for_hash()
    plt.savefig("{0}/C{1}-{2}.eps".format(folder, NB_CLIENTS, hash), format='eps')

    plt.show()

