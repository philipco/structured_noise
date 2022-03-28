"""
Created by Constantin Philippenko, 17th January 2022.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from src.CompressionModel import SQuantization, RandomSparsification, Sketching
from src.SyntheticDataset import SyntheticDataset

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

SIZE_DATASET = 2*10**3
DIM = 100
POWER_COV = 2
R_SIGMA=0

START_DIM = 10
END_DIM = 150

COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

USE_ORTHO_MATRIX = True


def compute_diag(dataset, compressor):

    X = dataset.X_complete

    X_compressed = X.copy()
    for i in range(SIZE_DATASET):
        X_compressed[i] = compressor.compress(X[i])

    cov_matrix = X_compressed.T.dot(X_compressed) / SIZE_DATASET

    return cov_matrix


def compute_trace(dataset: SyntheticDataset, dim: int) -> int:

    dataset.generate_constants(dim, size_dataset=SIZE_DATASET, power_cov=POWER_COV, r_sigma=R_SIGMA,
                       use_ortho_matrix=USE_ORTHO_MATRIX)
    dataset.define_compressors()
    dataset.generate_X()

    no_compressor = SQuantization(0, dim=dim)

    my_compressors = [no_compressor, dataset.quantizator, dataset.sparsificator, dataset.rand_sketcher]

    all_trace = []
    for compressor in my_compressors:
        cov_matrix = compute_diag(dataset, compressor)
        all_trace.append(np.trace(cov_matrix.dot(np.linalg.inv(dataset.upper_sigma))))

    return all_trace


def compute_theoretical_diag(dataset: SyntheticDataset):
    labels = ["no comprs.", "quantiz.", "rdk"]
    
    sigma_inv = np.linalg.inv(dataset.upper_sigma)

    ### No compression
    sigma = dataset.upper_sigma
    diag_sigma = np.diag(np.diag(sigma))
    all_covariance = [sigma.dot(sigma_inv)]

    ### Quantization
    cov_qtz = sigma - diag_sigma + np.sqrt(np.trace(sigma)) * np.sqrt(diag_sigma)
    all_covariance.append(cov_qtz.dot(sigma_inv))

    ### Sparsification
    ones = np.ones((dataset.dim, dataset.dim))
    p = dataset.sparsificator.sub_dim / dataset.dim
    P = p ** 2 * ones + (p - p ** 2) * np.eye(dataset.dim)
    cov_rdk = P * sigma / p ** 2
    all_covariance.append(cov_rdk.dot(sigma_inv))

    ### Sketching
    cov_sketching = sigma * (1 + 1 / dataset.sketcher.sub_dim) + np.trace(sigma) * np.identity(dataset.dim) / dataset.sketcher.sub_dim
    all_covariance.append(cov_sketching.dot(sigma_inv))

    # if USE_ORTHO_MATRIX:
    #     for i in range(len(all_covariance)):
    #         all_covariance[i] = dataset.ortho_matrix.T.dot(all_covariance[i]).dot(dataset.ortho_matrix)

    theoretical_trace = [np.trace(cov) for cov in all_covariance]
    return theoretical_trace


if __name__ == '__main__':

    print("Starting the script.")

    labels = ["no compr.", "quantiz.", "rdk", "rd gauss. proj."]
    theoretical_labels = ["no compr.", "quantiz.", "rdk", "gauss. proj."]

    range_trace = np.arange(START_DIM, END_DIM)

    trace_by_operators = [[] for i in range(len(labels))]
    theoretical_trace_by_operators = [[] for i in range(len(theoretical_labels))]

    for dim in range_trace:
        dataset = SyntheticDataset()
        all_trace = compute_trace(dataset, dim)
        for i in range(len(labels)):
            trace_by_operators[i].append(all_trace[i])
        all_theoretical_trace = compute_theoretical_diag(dataset)
        for i in range(len(theoretical_labels)):
            theoretical_trace_by_operators[i].append(all_theoretical_trace[i])


    fig, axes = plt.subplots(figsize=(10, 6))
    assert len(labels) == len(theoretical_labels), "Lenghts of empirical and theoretical traces are not identical."
    for i in range(len(labels)):
        axes.plot(np.log10(range_trace), np.log10(trace_by_operators[i]), label=labels[i], lw=2, color=COLORS[i])
        axes.plot(np.log10(range_trace), np.log10(theoretical_trace_by_operators[i]), label=theoretical_labels[i], lw=2,
                  color=COLORS[i], linestyle="--")

    # for ax in axes:
    # axes.set_ylim(top=4)
    axes.tick_params(axis='both', labelsize=15)
    axes.legend(loc='best', fontsize=15)
    axes.set_xlabel(r"$\log(i), \forall i \in \{1, ..., d\}$", fontsize=15)
    axes.title.set_text('Empirical vs theoretical trace')
    # axes[0].title.set_text('Theoretical eigenvalues')
    axes.set_ylabel(r"$\log(Trace(\frac{\mathcal C (X)^T.\mathcal C (X)}{n})_i)$", fontsize=15)

    print("Script completed.")

    plt.show()

