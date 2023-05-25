"""
Created by Constantin Philippenko, 1st April 2023.

Used to generate the figure in the paper which gives the trace w.r.t. the level of compression for both quantum and
cifar10, considering different pre-processing.
"""
from typing import List

import matplotlib
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm

from src.CompressionModel import Quantization, CompressionModel
from src.RealDataset import RealLifeDataset

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

COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:cyan"]

NB_CLIENTS = 1
OMEGA = 2

USE_ORTHO_MATRIX = False
HETEROGENEITY = "homog"

FONTSIZE = 16
LINESIZE = 4

densely_dotted = (0, (1, 1))


def compute_trace_for_various_preprocessing(dataset: RealLifeDataset, compressor: CompressionModel) \
        -> [float, float, float]:
    """Compute the trace of the compressed points' covariance for raw features, PCA features, normalized features."""

    X_compressed = dataset.X.copy()
    X_compressed_pca = dataset.X_pca.copy()
    X_compressed_raw = dataset.X_pca.copy()

    print(compressor.get_name())
    for i in tqdm(range(dataset.size_dataset), disable=DISABLE):
        X_compressed[i] = compressor.compress(dataset.X[i])
        X_compressed_pca[i] = compressor.compress(dataset.X_pca[i])
        X_compressed_raw[i] = compressor.compress(dataset.X_raw[i])

    cov_matrix = np.cov(X_compressed.T)
    cov_matrix_pca = np.cov(X_compressed_pca.T)
    cov_matrix_raw = np.cov(X_compressed_raw.T)

    trace = np.trace(cov_matrix.dot(dataset.upper_sigma_inv))
    print("Trace:", trace)
    trace_pca = np.trace(cov_matrix_pca.dot(dataset.upper_sigma_inv_pca))
    print("Trace PCA:", trace_pca)
    trace_raw = np.trace(cov_matrix_raw.dot(dataset.upper_sigma_inv_raw))
    print("Trace raw:", trace_raw)

    return trace, trace_pca, trace_raw, None


def compute_all_traces_for_various_preprocessing(dataset: RealLifeDataset) -> [List[float], List[float], List[float]]:
    """Compute the trace of the compressed points' covariance for each compressor, for raw features, PCA features, and
    normalized features."""

    no_compressor = Quantization(0, dim=dataset.dim)

    my_compressors = [no_compressor, dataset.quantizator, dataset.sparsificator, dataset.sketcher,
                      dataset.rand1, dataset.all_or_nothinger]

    all_traces_pca, all_traces, all_traces_no_std, all_traces_normalized = [], [], [], []
    for compressor in my_compressors:
        trace, trace_pca, trace_no_std, trace_normalized = compute_trace_for_various_preprocessing(dataset, compressor)
        all_traces_pca.append(trace_pca)
        all_traces.append(trace)
        all_traces_no_std.append(trace_no_std)
        all_traces_normalized.append(trace_normalized)

    return all_traces, all_traces_pca, all_traces_no_std, all_traces_normalized


if __name__ == '__main__':

    labels = ["no compr.", r"$s$-quantiz.", "sparsif.", "sketching",
              r"rand-$h$", "partial part."]



    datasets = ["quantum", "cifar10"]
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
            all_traces, all_traces_pca, all_traces_no_std, all_traces_normalized = \
                compute_all_traces_for_various_preprocessing(dataset)
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

