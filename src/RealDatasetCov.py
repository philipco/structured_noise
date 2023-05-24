"""
Created by Constantin Philippenko, 1st April 2023.

Used to generate the figure in the paper which gives the trace w.r.t. the level of compression for both quantum and
cifar10, considering different pre-processing.
"""

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm

from src.CompressionModel import Quantization
from src.RealDataset import RealLifeDataset
from src.utilities.PickleHandler import pickle_saver
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
DISABLE = True

#"#c08551",
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:cyan"]

NB_CLIENTS = 1
# DATASET_NAME = "Flowers102" # TODO Food101 : 1h
OMEGA = 2

BAR_PLOT = False

# In the case of heterogeneoux sigma and with non-diag H, check that random state of the orthogonal matrix is set to 5.
USE_ORTHO_MATRIX = False
HETEROGENEITY = "homog"

FONTSIZE = 16
LINESIZE = 4

densely_dotted = (0, (1, 1))


def compute_diag(dataset, compressor):

    X_compressed = dataset.X.copy()
    X_compressed_pca = dataset.X_pca.copy()
    X_compressed_raw = dataset.X_pca.copy()
    X_compressed_normalized = dataset.X_normalized.copy()

    print(compressor.get_name())
    for i in tqdm(range(dataset.size_dataset), disable=DISABLE):
        X_compressed[i] = compressor.compress(dataset.X[i])
        X_compressed_pca[i] = compressor.compress(dataset.X_pca[i])
        X_compressed_raw[i] = compressor.compress(dataset.X_raw[i])
        # X_compressed_normalized[i] = compressor.compress(dataset.X_normalized[i])

    cov_matrix = np.cov(X_compressed.T)
    cov_matrix_pca = np.cov(X_compressed_pca.T)
    cov_matrix_raw = np.cov(X_compressed_raw.T)
    # cov_matrix_normalized = np.cov(X_compressed_normalized.T)

    trace = np.trace(cov_matrix.dot(dataset.upper_sigma_inv))
    print("Trace:", trace)
    trace_pca = np.trace(cov_matrix_pca.dot(dataset.upper_sigma_inv_pca))
    print("Trace PCA:", trace_pca)
    trace_raw = np.trace(cov_matrix_raw.dot(dataset.upper_sigma_inv_raw))
    print("Trace raw:", trace_raw)
    trace_normalized = None # np.trace(cov_matrix_normalized.dot(dataset.upper_sigma_inv_normalized))
    # print("Trace normalized:", trace_normalized)

    return trace, trace_pca, trace_raw, trace_normalized


def compute_bound_qtzd(sigma, sigma_inv):
    residual1 = np.diag(np.diag(sigma)) @ sigma_inv
    residual2 = np.diag(np.sqrt(np.diag(sigma))) @ sigma_inv
    return dataset.dim + np.sqrt(np.trace(sigma)) * np.trace(residual2) - np.trace(residual1)


def compute_diag_matrices(dataset: RealLifeDataset):


    no_compressor = Quantization(0, dim=dataset.dim)

    my_compressors = [no_compressor, dataset.quantizator, dataset.sparsificator, dataset.sketcher,
                      dataset.rand1, dataset.all_or_nothinger]

    all_traces_pca, all_traces, all_traces_no_std, all_traces_normalized = [], [], [], []
    for compressor in my_compressors:
        trace, trace_pca, trace_no_std, trace_normalized = compute_diag(dataset, compressor)
        all_traces_pca.append(trace_pca)
        all_traces.append(trace)
        all_traces_no_std.append(trace_no_std)
        all_traces_normalized.append(trace_normalized)

    return all_traces, all_traces_pca, all_traces_no_std, all_traces_normalized




def plot_eigenvalues_and_compute_trace(dataset_name, labels, omega: int = OMEGA):
    dataset = RealLifeDataset(dataset_name, omega)
    all_diagonals, all_traces, labels, dataset = compute_diag_matrices(dataset, labels=labels)

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

    labels = ["no compr.", r"$s$-quantiz.", "sparsif.", "sketching",
              r"rand-$h$", "partial part."]

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
            fig, axes = plt.subplots(3, 1, figsize=(8, 9))
            traces_by_omega = dict(zip(labels, [[] for i in range(len(labels))]))
            traces_by_omega_pca = dict(zip(labels, [[] for i in range(len(labels))]))
            traces_by_omega_raw = dict(zip(labels, [[] for i in range(len(labels))]))
            traces_by_omega_normalized = dict(zip(labels, [[] for i in range(len(labels))]))
            squantization = [32, 16, 8, 6, 4, 2, 1]
            real_omegas = []
            for s in squantization:
                dataset = RealLifeDataset(dataset_name, s)
                real_omegas.append(dataset.quantizator.omega_c)
                all_traces, all_traces_pca, all_traces_no_std, all_traces_normalized = compute_diag_matrices(dataset)
                for label, value, value_pca, value_no_std, value_normalized in zip(labels, all_traces, all_traces_pca,
                                                                                   all_traces_no_std,
                                                                                   all_traces_normalized):
                    traces_by_omega[label].append(value)
                    traces_by_omega_pca[label].append(value_pca)
                    traces_by_omega_raw[label].append(value_no_std)
                    traces_by_omega_normalized[label].append(value_normalized)

            linestyle = ["-", "-.", "--", densely_dotted]
            markers = ["h", "X", "P", "*"]
            lines_labels = ["w. std", "w. pca", "raw data"]
            traces =  [traces_by_omega, traces_by_omega_pca, traces_by_omega_raw]

            for i in range(len(labels)):
                for j in range(len(axes)):
                    axes[j].plot(real_omegas, traces[j][labels[i]], lw=LINESIZE, linestyle=linestyle[j], label=labels[i],
                                 color=COLORS[i], alpha=0.7, fillstyle='full')
                    axes[j].plot(real_omegas, traces[j][labels[i]], marker=markers[j], ms=8, linestyle="None",
                                 color=COLORS[i])
                # axes[0].plot(real_omegas, traces_by_omega[labels[i]], lw=LINESIZE, linestyle="-.",
                #         color=COLORS[i], alpha=0.7, fillstyle='full')
                # axes[0].plot(real_omegas, traces_by_omega[labels[i]], marker="X", ms=8, linestyle="None",
                #         color=COLORS[i])
                # axes[1].plot(real_omegas, traces_by_omega_pca[labels[i]], lw=LINESIZE, linestyle="--",
                #         color=COLORS[i], alpha=0.7, fillstyle='full')
                # axes[1].plot(real_omegas, traces_by_omega_pca[labels[i]], marker="P", ms=8, linestyle="None",
                #              color=COLORS[i])
                # axes[3].plot(real_omegas, traces_by_omega_normalized[labels[i]], lw=LINESIZE, linestyle=densely_dotted,
                #              color=COLORS[i], alpha=0.7, fillstyle='full')
                # axes[3].plot(real_omegas, traces_by_omega_normalized[labels[i]], marker="*", ms=10, linestyle="None",
                #              color=COLORS[i])



            axes[0].get_xaxis().set_visible(False)
            axes[1].get_xaxis().set_visible(False)
            axes[1].set_ylabel(r"$\log(\mathrm{Tr}(\mathfrak{C}^{\mathrm{ania}} H^{-1}))$", fontsize=FONTSIZE)
            axes[2].set_xlabel(r"Value of $\omega$", fontsize=FONTSIZE)

            for ax in axes:
                ax.set_yscale('log')
                ax.set_xscale('log')
                ax.tick_params(axis='both', labelsize=FONTSIZE)
                ax.grid(True)
            axes[0].set_ylim(top=traces_by_omega[labels[3]][-1]*1.2)
            axes[1].set_ylim(top=traces_by_omega[labels[3]][-1] * 1.2)

            legend_line = [Line2D([0], [0], linestyle=linestyle[i], color="black", lw=2, label=lines_labels[i],
                                  marker=markers[i], ms=8) for i in range(len(axes))]

            l1 = axes[0].legend(loc="upper left", fontsize=FONTSIZE, framealpha=0.4)
            l2 = axes[1].legend(handles=legend_line, loc="upper left", fontsize=FONTSIZE, framealpha=0.4)
            axes[0].add_artist(l1)
            axes[1].add_artist(l2)


            plt.subplots_adjust(wspace=0, hspace=0) # To remove the space between subplots
            plt.savefig("{0}/omega_{1}.pdf".format(FOLDER, dataset_name), bbox_inches='tight', dpi=600)
            plt.close()



