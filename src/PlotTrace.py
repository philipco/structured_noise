"""
Created by Constantin Philippenko, 17th January 2022.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from src.CompressionModel import SQuantization, RandomSparsification, Sketching
from src.SyntheticDataset import SyntheticDataset
from src.TheoreticalCov import compute_theoretical_trace
from src.Utilities import create_folder_if_not_existing

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

SIZE_DATASET = 10**3
DIM = 100
POWER_COV = 4
R_SIGMA=0

START_DIM = 2
END_DIM = 50


COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:brown", "tab:purple", "tab:cyan"]

USE_ORTHO_MATRIX = False
HETEROGENEITY = "homog" # "wstar" "sigma" "homog"


def compute_diag(dataset, compressor):

    X = dataset.X_complete

    X_compressed = X.copy()
    for i in range(SIZE_DATASET):
        X_compressed[i] = compressor.compress(X[i])

    cov_matrix = X_compressed.T.dot(X_compressed) / SIZE_DATASET

    return cov_matrix


def compute_trace(dataset: SyntheticDataset, dim: int) -> int:

    dataset.generate_constants(dim, size_dataset=SIZE_DATASET, power_cov=POWER_COV, r_sigma=R_SIGMA,
                               use_ortho_matrix=USE_ORTHO_MATRIX, heterogeneity=HETEROGENEITY)
    dataset.define_compressors()
    dataset.generate_X()

    no_compressor = SQuantization(0, dim=dim)

    my_compressors = [no_compressor, dataset.quantizator, dataset.sparsificator, dataset.rand_sketcher,
                      dataset.rand1, dataset.all_or_nothinger]

    all_trace = []
    for compressor in my_compressors:
        cov_matrix = compute_diag(dataset, compressor)
        all_trace.append(np.trace(cov_matrix.dot(np.linalg.inv(dataset.upper_sigma))))

    return all_trace


if __name__ == '__main__':

    print("Starting the script.")

    labels = ["no compr.", "quantiz.", "rdk", "gauss. proj.", "rand1", "all or noth."]

    range_trace = np.arange(START_DIM, END_DIM)

    trace_by_operators = [[] for i in range(len(labels))]
    theoretical_trace_by_operators = [[] for i in range(len(labels))]

    for dim in range_trace:
        dataset = SyntheticDataset()
        all_trace = compute_trace(dataset, dim)
        for i in range(len(labels)):
            trace_by_operators[i].append(all_trace[i])
        all_theoretical_trace = [compute_theoretical_trace(dataset, "No compression"),
                                 compute_theoretical_trace(dataset, "Qtzd"),
                                 compute_theoretical_trace(dataset, "Sparsification"),
                                 compute_theoretical_trace(dataset, "Sketching"),
                                 compute_theoretical_trace(dataset, "Rand1"),
                                 compute_theoretical_trace(dataset, "AllOrNothing")]
        for i in range(len(labels)):
            theoretical_trace_by_operators[i].append(all_theoretical_trace[i])


    fig, axes = plt.subplots(figsize=(7, 6))
    for i in range(len(labels)):
        axes.plot(np.log10(range_trace), np.log10(trace_by_operators[i]), label=labels[i], lw=2, color=COLORS[i])
        axes.plot(np.log10(range_trace), np.log10(theoretical_trace_by_operators[i]), label=labels[i], lw=2,
                  color=COLORS[i], linestyle="--")

    axes.tick_params(axis='both', labelsize=15)
    axes.legend(loc='best', fontsize=15)
    axes.set_xlabel(r"$\log(i), \forall i \in \{1, ..., d\}$", fontsize=15)
    axes.title.set_text('Empirical (plain) vs theoretical trace (dashed)')
    axes.set_ylabel(r"$\log(Trace(\frac{\mathcal C (X)^T.\mathcal C (X)}{n})_i)$", fontsize=15)

    print("Script completed.")
    folder = "pictures/trace/"
    create_folder_if_not_existing(folder)

    hash = "N{0}-P{1}".format(SIZE_DATASET, POWER_COV)
    if USE_ORTHO_MATRIX:
        hash = "{0}-ortho".format(hash)
    plt.savefig("{0}/{1}.eps".format(folder, hash), format='eps')

    plt.show()

