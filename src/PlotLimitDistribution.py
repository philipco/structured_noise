"""Created by Constantin Philippenko, 11th April 2022."""
import math
import random

import numpy as np
from PIL import Image
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

from PlotUtils import confidence_ellipse
from src.CompressionModel import *
from src.SGD import SGDCompressed
from src.SyntheticDataset import SyntheticDataset
from src.Utilities import create_folder_if_not_existing

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

SIZE_DATASET = 1000
POWER_COV = 2
R_SIGMA = 0

DIM = 2

USE_ORTHO_MATRIX = True
DO_LOGISTIC_REGRESSION = False

NB_REMOVED_POINTS = 50

COLORS = ["tab:blue", "tab:orange", "tab:brown", "tab:green", "tab:red", "tab:purple", "tab:cyan"]



def plot_limit_distribution(sigma, covariances, all_sgd, compressors, labels):

    inv_sigma = np.linalg.inv(sigma)
    limit_sigma_no_compr = inv_sigma @ covariances[0] @ inv_sigma / SIZE_DATASET
    eig, _ = np.linalg.eig(limit_sigma_no_compr)
    ax_max = np.max([np.linalg.norm(x, ord=2) for x in all_sgd[0]])
    print(ax_max)

    cov_matrix_no_compr_limit_distrib = all_sgd[0][NB_REMOVED_POINTS:].T.dot(all_sgd[0][NB_REMOVED_POINTS:]) / (SIZE_DATASET - NB_REMOVED_POINTS)

    fig_distrib, axes_distrib = plt.subplots(2, 3, figsize=(10, 8))
    axes_distrib = axes_distrib.flat
    for i in range(len(covariances) - 1):
        axes_distrib[i].axvline(c='grey', lw=1)
        axes_distrib[i].axhline(c='grey', lw=1)

        limit_sigma = inv_sigma @ covariances[i+1] @ inv_sigma / SIZE_DATASET
        cov_matrix_limit_distrib = all_sgd[i+1][NB_REMOVED_POINTS:].T.dot(all_sgd[i+1][NB_REMOVED_POINTS:]) / (SIZE_DATASET - NB_REMOVED_POINTS)
        print(cov_matrix_limit_distrib)
        axes_distrib[i].scatter(all_sgd[i+1][:, 0], all_sgd[i+1][:, 1], alpha=0.5, color=COLORS[1], s=10, label=r"$\bar w - w_*$")
        confidence_ellipse(limit_sigma, "$\Sigma^{-1} C \Sigma^{-1} / n$", axes_distrib[i], edgecolor=COLORS[1], zorder=0)
        confidence_ellipse(cov_matrix_limit_distrib, r"$\mathrm{Cov}(\bar w - w_*)$", axes_distrib[i], edgecolor=COLORS[1],
                           zorder=0, linestyle="--")

        axes_distrib[i].scatter(all_sgd[0][:, 0], all_sgd[0][:, 1], alpha=0.5, color=COLORS[0], s=10, label="no compr.")
        confidence_ellipse(limit_sigma_no_compr, "$\Sigma^{-1} C \Sigma^{-1} / n$", axes_distrib[i], edgecolor=COLORS[0], zorder=0)
        confidence_ellipse(cov_matrix_no_compr_limit_distrib, r"$\mathrm{Cov}(\bar w - w_*)$", axes_distrib[i], edgecolor=COLORS[0],
                           zorder=0, linestyle="--")

        # axes_distrib[i].set_xlim(-ax_max, ax_max)
        # axes_distrib[i].set_ylim(-ax_max, ax_max)
        axes_distrib[i].scatter(0, 0, c='red', s=3)

        axes_distrib[i].set_title(compressors[i+1].get_name())
    axes_distrib[0].legend(fancybox=True, framealpha=0.5)
    axes_distrib[0].axis('equal')

    # fig.subplots_adjust(hspace=0.25)
    filename = "pictures/limit_distribution/ellipse"
    if USE_ORTHO_MATRIX:
        filename = "{0}-ortho".format(filename)
    plt.savefig("{0}.png".format(filename), bbox_inches='tight', dpi=600)


def compute_covariance(dataset, compressor):
    X = dataset.X_complete

    X_compressed = X.copy()

    for i in range(SIZE_DATASET):
        X_compressed[i] = compressor.compress(X[i])

    cov_matrix = X_compressed.T.dot(X_compressed) / SIZE_DATASET

    return cov_matrix


def get_all_covariances(dataset: SyntheticDataset, my_compressors):
    all_covariances = []
    for compressor in tqdm(my_compressors):
        cov_matrix = compute_covariance(dataset, compressor)
        all_covariances.append(cov_matrix)

    return all_covariances

if __name__ == '__main__':

    folder = "pictures/limit_distribution"
    create_folder_if_not_existing(folder)

    synthetic_dataset = SyntheticDataset()
    synthetic_dataset.generate_dataset(DIM, size_dataset=SIZE_DATASET, power_cov=POWER_COV, r_sigma=R_SIGMA,
                                       use_ortho_matrix=USE_ORTHO_MATRIX, do_logistic_regression=DO_LOGISTIC_REGRESSION,
                                       eigenvalues=np.array([1,10]))
    # synthetic_dataset = SyntheticDataset()
    # synthetic_dataset.generate_constants(DIM, SIZE_DATASET, POWER_COV, R_SIGMA, USE_ORTHO_MATRIX,
    #                                      eigenvalues=np.array([0.001, 10]))
    # synthetic_dataset.define_compressors()
    # synthetic_dataset.generate_X()

    labels = ["no compr.", "quantiz.", "stab. quantiz.", "gauss. proj.", "sparsif", "rand1", "all or noth."]
    no_compressor = SQuantization(0, dim=DIM)
    my_compressors = [no_compressor, synthetic_dataset.quantizator, synthetic_dataset.stabilized_quantizator,
                      synthetic_dataset.rand_sketcher,
                      synthetic_dataset.sparsificator, synthetic_dataset.rand1,
                      synthetic_dataset.all_or_nothinger]

    all_covariances = get_all_covariances(synthetic_dataset, my_compressors)

    all_sgd = []
    for compressor in tqdm(my_compressors):
        sgd = SGDCompressed(synthetic_dataset, compressor).gradient_descent(label=compressor.get_name())
        all_sgd.append(sgd.all_avg_w - synthetic_dataset.w_star)

    # print(all_covariances)
    plot_limit_distribution(synthetic_dataset.upper_sigma, all_covariances, all_sgd, my_compressors, labels)