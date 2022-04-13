"""Created by Constantin Philippenko, 11th April 2022."""
import math
import random

import numpy as np
from PIL import Image
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from tqdm import tqdm

from PlotUtils import confidence_ellipse
from src.CompressionModel import *
from src.SGD import SGDCompressed
from src.SyntheticDataset import SyntheticDataset
from src.Utilities import create_folder_if_not_existing
from src.main import plot_only_avg

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

SIZE_DATASET = 10**5
POWER_COV = 2
R_SIGMA = 0

DIM = 2
EIGENVALUES = np.array([1,5])

USE_ORTHO_MATRIX = True
DO_LOGISTIC_REGRESSION = False

LAST_POINTS = 0

COLORS = ["tab:blue", "tab:orange", "tab:brown", "tab:green", "tab:red", "tab:purple", "tab:cyan"]


def compute_inversion(matrix, gamma0):
    return np.linalg.inv(matrix) #np.linalg.inv(2 * gamma0 * matrix - np.eye(2))


def compute_limit_distrib(inversed_matrix, error_cov, gamma0):
    return inversed_matrix @ error_cov @ inversed_matrix # gamma0**2 * inversed_matrix * error_cov


def compute_empirical_covariance(distance_to_opt):
    distance_to_opt = distance_to_opt[LAST_POINTS:]
    return distance_to_opt.T.dot(distance_to_opt) / len(distance_to_opt)


def get_legend_limit_distrib():
    legend_color = [Line2D([0], [0], color=COLORS[0], lw=2, label='no compr.'),
                    Line2D([0], [0], color=COLORS[1], lw=2, label='compr.')]

    legend_line = [Line2D([0], [0], linestyle="--", color="black", lw=1, label=r"$\mathrm{Cov}( \sqrt{n} (\bar w - w_*))$"),
                   Line2D([0], [0], linestyle="-", color="black", lw=1,
                          label=r"$\Sigma^{-1} \mathrm{Cov}(\mathcal C (x)) \Sigma^{-1}$"),
                   Line2D([0], [0], linestyle=":", color="black", lw=1, label=r"$\Sigma^{-1} C \Sigma^{-1}$"),
                   Line2D([], [], color='black', marker="*", linestyle='None',
                          markersize=4, label=r"last $\bar w - w_*$")
                   ]
    return legend_line, legend_color


def plot_limit_distribution(sigma, covariances, cov_grad_error, all_sgd, compressors, labels):

    gamma0 = 1 / (2*min(EIGENVALUES))
    inversed_matrix = compute_inversion(sigma, gamma0)

    fig_distrib, axes_distrib = plt.subplots(2, 3, figsize=(10, 8))
    axes_distrib = axes_distrib.flat
    for i in range(len(covariances) - 1):
        axes_distrib[i].axvline(c='grey', lw=1)
        axes_distrib[i].axhline(c='grey', lw=1)

        axes_distrib[i].plot(all_sgd[i+1][-1:, 0], all_sgd[i+1][-1:, 1], marker = "*", color=COLORS[1], markersize=10)
        confidence_ellipse(compute_limit_distrib(inversed_matrix, covariances[i+1], gamma0), "",
                           axes_distrib[i], edgecolor=COLORS[1], zorder=0, lw=2)
        confidence_ellipse(compute_limit_distrib(inversed_matrix, cov_grad_error[i+1], gamma0), "",
                           axes_distrib[i], edgecolor=COLORS[1], zorder=0, linestyle=":", lw=2)
        confidence_ellipse(compute_empirical_covariance(all_sgd[i + 1]), "", axes_distrib[i], edgecolor=COLORS[1],
                           zorder=0, linestyle="--", lw=2)

        axes_distrib[i].plot(all_sgd[0][-1, 0], all_sgd[0][-1, 1], marker="*", color=COLORS[0], markersize=10)
        confidence_ellipse(compute_limit_distrib(inversed_matrix, covariances[0], gamma0), "",
                           axes_distrib[i], edgecolor=COLORS[0], zorder=0, lw=2)
        confidence_ellipse(compute_limit_distrib(inversed_matrix, cov_grad_error[0], gamma0), "",
                           axes_distrib[i], edgecolor=COLORS[0], zorder=0, linestyle=":", lw=2)
        confidence_ellipse(compute_empirical_covariance(all_sgd[0]), "", axes_distrib[i],
                           edgecolor=COLORS[0], zorder=0, linestyle="--", lw=2)

        axes_distrib[i].scatter(0, 0, c='red', s=3)
        axes_distrib[i].set_title(compressors[i+1].get_name())

    legend_line, legend_color = get_legend_limit_distrib()

    first_legend = axes_distrib[0].legend(handles=legend_color, loc='lower right', fancybox=True, framealpha=0.5)

    axes_distrib[0].add_artist(first_legend)
    axes_distrib[0].legend(handles=legend_line, loc='upper right', fancybox=True, framealpha=0.5)
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
                                       eigenvalues=EIGENVALUES)

    labels = ["no compr.", "quantiz.", "stab. quantiz.", "gauss. proj.", "sparsif", "rand1", "all or noth."]
    no_compressor = SQuantization(0, dim=DIM)
    my_compressors = [no_compressor, synthetic_dataset.quantizator, synthetic_dataset.stabilized_quantizator,
                      synthetic_dataset.rand_sketcher,
                      synthetic_dataset.sparsificator, synthetic_dataset.rand1,
                      synthetic_dataset.all_or_nothinger]

    all_covariances = get_all_covariances(synthetic_dataset, my_compressors)

    all_sgd_descent = []
    all_avg_sgd = []
    all_cov_grad_error = []
    for compressor in my_compressors:
        sgd = SGDCompressed(synthetic_dataset, compressor).gradient_descent(label=compressor.get_name(),
                                                                            deacreasing_step_size=True)
        all_sgd_descent.append(sgd)
        all_avg_sgd.append((sgd.all_avg_w - synthetic_dataset.w_star) * np.sqrt(SIZE_DATASET))
        all_cov_grad_error.append(sgd.cov_grad_error)

    plot_only_avg(all_sgd_descent[1:], sgd_nocompr=all_sgd_descent[0], optimal_loss=0,
                  hash_string="limit_distribution/avg_sgd")

    plot_limit_distribution(synthetic_dataset.upper_sigma, all_covariances, all_cov_grad_error, all_avg_sgd,
                            my_compressors, labels)