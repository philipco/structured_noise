"""
Created by Constantin Philippenko, 17th January 2022.
"""
import cmath
from typing import List

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.CompressionModel import Quantization
from src.RealDataset import RealLifeDataset
from src.SyntheticDataset import SyntheticDataset
from src.TheoreticalCov import get_theoretical_cov
from src.Utilities import create_folder_if_not_existing
from src.federated_learning.Client import Client

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})


sns.set(font='serif', style='white',
        palette="tab10",
        font_scale=1.2,
        rc={'text.usetex': True, 'pgf.rcfonts': False})



NB_CLIENTS = 1
# DATASET_NAME = "Flowers102" # TODO Food101 : 1h
OMEGA = 30

# In the case of heterogeneoux sigma and with non-diag H, check that random state of the orthogonal matrix is set to 5.
USE_ORTHO_MATRIX = False
HETEROGENEITY = "homog"

FONTSIZE = 20
LINESIZE = 4


def compute_diag(dataset, compressor):

    X = dataset.X_complete

    X_compressed = X.copy()

    print(compressor.get_name())
    for i in tqdm(range(dataset.size_dataset)):
        X_compressed[i] = compressor.compress(X[i])

    cov_matrix = np.cov(X_compressed.T)

    trace = np.trace(cov_matrix.dot(dataset.upper_sigma_inv))
    print("Trace:", trace)

    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    return eigenvalues, cov_matrix, trace


def compute_diag_matrices(dataset: RealLifeDataset, labels):


    no_compressor = Quantization(0, dim=dataset.dim)

    my_compressors = [no_compressor, dataset.quantizator, dataset.sparsificator, dataset.sketcher, dataset.rand1,
                      dataset.all_or_nothinger]

    all_diagonals, all_traces = [], []
    for compressor in my_compressors:
        diag, cov_matrix, trace = compute_diag(dataset, compressor)
        all_diagonals.append(diag)
        all_traces.append(trace)

    return all_diagonals, all_traces, labels, dataset


def compute_theoretical_diag(dataset: SyntheticDataset, labels):

    ### No compression
    nb_clients = 1
    all_covariance = [get_theoretical_cov(dataset, nb_clients, "No compression"),
                      get_theoretical_cov(dataset, nb_clients, "Qtzd"),
                      get_theoretical_cov(dataset, nb_clients, "Sparsification"),
                      get_theoretical_cov(dataset, nb_clients, "Sketching"),
                      get_theoretical_cov(dataset, nb_clients, "Rand1"),
                      get_theoretical_cov(dataset, nb_clients, "PartialParticipation")]


    all_diagonals = [np.diag(cov) for cov in all_covariance]
    return all_diagonals, labels

def plot_eigenvalues_and_compute_trace(dataset_name, labels):
    dataset = RealLifeDataset(dataset_name, OMEGA)
    all_diagonals, all_traces, labels, dataset = compute_diag_matrices(dataset, labels=labels)
    # all_theoretical_diagonals, theoretical_labels = compute_theoretical_diag(dataset, labels=labels)

    fig, axes = plt.subplots(1, 1, figsize=(6, 6))
    for (diagonal, label) in zip(all_diagonals, labels):
        axes.plot(np.log10(np.arange(1, dataset.dim + 1)), np.sort(np.log10(diagonal))[::-1], label=label, lw=LINESIZE)
    # for (diagonal, label) in zip(all_theoretical_diagonals, theoretical_labels):
    #     axes[1].plot(np.log10(np.arange(1, dataset.dim + 1)), np.sort(np.log10(diagonal))[::-1], label=label, lw = LINESIZE, linestyle="--")

    # for ax in axes:
    axes.tick_params(axis='both', labelsize=FONTSIZE)
    axes.legend(loc='lower left', fontsize=FONTSIZE)
    axes.set_xlabel(r"$\log(i), \forall i \in \{1, ..., d\}$", fontsize=FONTSIZE)
    axes.set_title('Empirical eigenvalues', fontsize=FONTSIZE)
    # axes[1].set_title('Theoretical eigenvalues', fontsize=FONTSIZE)
    axes.set_ylabel(r"$\log(\mathrm{eig}(\mathfrak{C}^{\mathrm{ania}})_i)$", fontsize=FONTSIZE)
    plt.legend(loc='lower left', fontsize=FONTSIZE)
    folder = "pictures/epsilon_eigenvalues/{0}".format(dataset_name)
    create_folder_if_not_existing(folder)

    hash = dataset.string_for_hash(nb_runs=1)
    plt.savefig("{0}/C{1}-{2}.pdf".format(folder, NB_CLIENTS, hash), bbox_inches='tight', dpi=600)
    plt.close()
    return all_traces

def plot_traces_by_dataset(all_traces, compressors):

    df = pd.DataFrame(all_traces, index=compressors)
    df.index.name = "dataset"
    df = df.stack().reset_index()
    df.columns = ["compressor", "dataset", "trace"]

    # Create a stacked bar plot
    sns.set(style="whitegrid")
    sns.barplot(x="dataset", y="trace", hue="compressor", data=df, palette="tab10", edgecolor="none")
    plt.yscale('log')
    folder = "pictures/epsilon_eigenvalues/"
    create_folder_if_not_existing(folder)
    plt.savefig("{0}/trace_barplot_real_datasets.pdf".format(folder, NB_CLIENTS, hash), bbox_inches='tight', dpi=600)
    plt.show()




if __name__ == '__main__':

    labels = ["no compr.", "1-quantiz.", "sparsif.", "sketching", "rand-1", "partial part."]
    datasets = ["mnist", "fashion_mnist", "emnist", "cifar10", "cifar100", "EuroSAT"] #"mnist", "cifar10", "Flowers102"]
    traces = {}
    for dataset_name in datasets:
        all_traces = plot_eigenvalues_and_compute_trace(dataset_name, labels)
        traces[dataset_name] = all_traces

    plot_traces_by_dataset(traces, labels)
