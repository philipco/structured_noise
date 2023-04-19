"""
Created by Constantin Philippenko, 17th January 2022.
"""
import copy

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm

from src.CompressionModel import Quantization
from src.utilities.PickleHandler import pickle_saver
from src.RealDataset import RealLifeDataset
from src.SyntheticDataset import SyntheticDataset
from src.TheoreticalCov import get_theoretical_cov
from src.utilities.Utilities import create_folder_if_not_existing

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

FOLDER = "pictures/real_dataset"

COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:brown", "tab:purple", "tab:cyan"]

NB_CLIENTS = 1
# DATASET_NAME = "Flowers102" # TODO Food101 : 1h
OMEGA = 2

BAR_PLOT = False

# In the case of heterogeneoux sigma and with non-diag H, check that random state of the orthogonal matrix is set to 5.
USE_ORTHO_MATRIX = False
HETEROGENEITY = "homog"

FONTSIZE = 16
LINESIZE = 4


def compute_diag(dataset, compressor):

    X = dataset.X_complete
    X_pca = dataset.X_pca

    X_compressed = dataset.X_complete.copy()
    X_compressed_pca = dataset.X_pca.copy()

    print(compressor.get_name())
    for i in tqdm(range(10000)):
        X_compressed[i] = compressor.compress(X[i])
        X_compressed_pca[i] = compressor.compress(X_pca[i])

    cov_matrix = np.cov(X_compressed.T)
    cov_matrix_pca = np.cov(X_compressed_pca.T)

    # plt.imshow(cov_matrix)
    # plt.show()

    trace = np.trace(cov_matrix.dot(dataset.upper_sigma_inv))
    print("Trace:", trace)
    trace_pca = np.trace(cov_matrix_pca.dot(dataset.upper_sigma_inv_pca))
    print("Trace PCA:", trace_pca)
    # print("Trace rd-k from formula:", np.trace(np.diag(np.diag(dataset.upper_sigma_pca)).dot(dataset.upper_sigma_inv_pca)) / dataset.rand1.level)

    return trace, trace_pca


def compute_diag_matrices(dataset: RealLifeDataset, labels):


    no_compressor = Quantization(0, dim=dataset.dim)

    my_compressors = [no_compressor, dataset.quantizator, dataset.sparsificator, dataset.sketcher,
                      dataset.rand1, dataset.all_or_nothinger]

    all_traces_pca, all_traces = [], []
    for compressor in my_compressors:
        trace, trace_pca = compute_diag(dataset, compressor)
        all_traces_pca.append(trace_pca)
        all_traces.append(trace)

    return all_traces, all_traces_pca




def plot_eigenvalues_and_compute_trace(dataset_name, labels, omega: int = OMEGA):
    dataset = RealLifeDataset(dataset_name, omega)
    all_diagonals, all_traces, labels, dataset = compute_diag_matrices(dataset, labels=labels)
    # all_theoretical_diagonals, theoretical_labels = compute_theoretical_diag(dataset, labels=labels)

    fig, axes = plt.subplots(1, 1, figsize=(6, 6))
    for (diagonal, label) in zip(all_diagonals, labels):
        axes.plot(np.arange(1, dataset.dim + 1), np.sort(np.log10(diagonal))[::-1], label=label, lw=LINESIZE)

    axes.tick_params(axis='both', labelsize=FONTSIZE)
    axes.legend(loc='lower left', fontsize=FONTSIZE)
    axes.set_xlabel(r"$\log(i), \forall i \in \{1, ..., d\}$", fontsize=FONTSIZE)
    axes.set_title('Empirical eigenvalues', fontsize=FONTSIZE)
    axes.set_ylabel(r"$\log(\mathrm{eig}(\mathfrak{C}^{\mathrm{ania}})_i)$", fontsize=FONTSIZE)
    plt.legend(loc='lower left', fontsize=FONTSIZE)
    folder = "{0}/{1}".format(FOLDER, dataset_name)
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
    sns.set(rc={'figure.figsize': (10, 5)})
    sns.set(style="whitegrid")
    sns.barplot(x="dataset", y="trace", hue="compressor", data=df, palette="tab10", edgecolor="none")
    plt.yscale('log')
    plt.ylabel(r"$\log(\mathrm{Tr}(\mathfrak{C}^{\mathrm{ania}} H^{-1}))$")
    plt.xlabel("")
    create_folder_if_not_existing(FOLDER)
    plt.savefig("{0}/trace_barplot_real_datasets.pdf".format(FOLDER, NB_CLIENTS, hash), bbox_inches='tight', dpi=600)
    plt.show()




if __name__ == '__main__':

    labels = ["no compr.", "s-quantiz.", "sparsif.", "sketching",
              "rand-k", "partial part."]

    if BAR_PLOT:

        datasets = ["cifar10", "mnist", "fashion-mnist", "emnist", "flowers102", "euroSAT"] #"emnist", "cifar10", "cifar100"]#, "EuroSAT"] #"mnist", "cifar10", "Flowers102"]
        traces = {}
        for dataset_name in datasets:
            all_traces = plot_eigenvalues_and_compute_trace(dataset_name, labels)
            traces[dataset_name] = all_traces

        create_folder_if_not_existing(FOLDER)
        pickle_saver(traces, "{0}/all_traces.pkl".format(FOLDER))

        plot_traces_by_dataset(traces, labels)

    else:

        datasets = ["quantum", "cifar10"] #, "mnist", "fashion-mnist", "emnist", "flowers102",
                    # "euroSAT"]
        for dataset_name in datasets:
            print(">>>>> {0}".format(dataset_name))
            fig, axes = plt.subplots(2, 1, figsize=(8, 6))
            traces_by_omega = dict(zip(labels, [[] for i in range(len(labels))]))
            traces_by_omega_pca = dict(zip(labels, [[] for i in range(len(labels))]))
            squantization = [32, 16, 8, 6, 4, 2, 1]
            real_omegas = []
            for s in squantization:
                dataset = RealLifeDataset(dataset_name, s)
                real_omegas.append(dataset.quantizator.omega_c)
                all_traces, all_traces_pca = compute_diag_matrices(dataset, labels=labels)
                for label, value, value_pca in zip(labels, all_traces, all_traces_pca):
                    traces_by_omega[label].append(value)
                    traces_by_omega_pca[label].append(value_pca)

            # ax.plot(real_omegas, traces_by_omega[labels[0]], label=labels[0], lw=LINESIZE,
            #         color=COLORS[0], alpha=0.7, fillstyle='full')
            # ax.plot(real_omegas, traces_by_omega[labels[0]], marker="h", ms=8, linestyle="None",
            #         color=COLORS[0])
            for i in range(len(labels)):
                axes[0].plot(real_omegas, traces_by_omega[labels[i]], label=labels[i], lw=LINESIZE,
                        color=COLORS[i], alpha=0.7, fillstyle='full')
                axes[0].plot(real_omegas, traces_by_omega[labels[i]], marker="h", ms=8, linestyle="None",
                        color=COLORS[i])
                axes[1].plot(real_omegas, traces_by_omega_pca[labels[i]], lw=LINESIZE, linestyle="--",
                        color=COLORS[i], alpha=0.7, fillstyle='full')
                axes[1].plot(real_omegas, traces_by_omega_pca[labels[i]], marker="P", ms=8, linestyle="None",
                             color=COLORS[i])


            axes[0].get_xaxis().set_visible(False)
            axes[1].set_xlabel(r"Value of $\omega$", fontsize=FONTSIZE)
            axes[0].set_ylabel(r"$\log(\mathrm{Tr}(\mathfrak{C}^{\mathrm{ania}} H^{-1}))$", fontsize=FONTSIZE)
            axes[1].set_ylabel(r"$\log(\mathrm{Tr}(\mathfrak{C}^{\mathrm{ania}} H^{-1}))$", fontsize=FONTSIZE)

            for ax in axes:
                ax.set_yscale('log')
                ax.set_xscale('log')
                ax.tick_params(axis='both', labelsize=FONTSIZE)
                ax.grid(True)
                ax.set_ylim(top=traces_by_omega[labels[2]][-1]*1.2)

            legend_line = [Line2D([0], [0], color="black", lw=2, label='w.o. pca', marker="h"),
                           Line2D([0], [0], linestyle="--", color="black", lw=2, label='w. pca', marker="P")]

            l1 = axes[0].legend(loc="upper left", fontsize=FONTSIZE)
            l2 = axes[1].legend(handles=legend_line, loc="upper left", fontsize=FONTSIZE)
            axes[0].add_artist(l1)
            axes[1].add_artist(l2)


            plt.subplots_adjust(wspace=0, hspace=0) # To remove the space between subplots
            plt.savefig("{0}/omega_{1}.pdf".format(FOLDER, dataset_name), bbox_inches='tight', dpi=600)
            plt.close()



