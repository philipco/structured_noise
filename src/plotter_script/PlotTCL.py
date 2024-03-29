"""Created by Constantin Philippenko, 09th August 2022.

Useless code kept in case of."""
from typing import List

import sympy as sy

from src.federated_learning.Client import Client, check_clients

sy.init_printing(use_unicode=True)

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm

from src.utilities.PlotUtils import plot_ellipse
from src.CompressionModel import *
from src.SGD import SGDArtemis, SeriesOfSGD
from src.TheoreticalCov import get_theoretical_cov, compute_inversion, compute_empirical_covariance, \
    compute_limit_distrib
from src.utilities.Utilities import create_folder_if_not_existing
from src.main import plot_only_avg

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

SIZE_DATASET = 10**6
POWER_COV = 4
R_SIGMA = 0

DIM = 2
EIGENVALUES = np.array([1,0.2])


FONTSIZE = 17
LINESIZE = 2

NB_CLIENTS = 10
HETEROGENEITY = "wstar" # "wstar" "sigma" "homog"

USE_ORTHO_MATRIX = True
DO_LOGISTIC_REGRESSION = False

LAST_POINTS = 0
NB_TRY = 100

step_size = lambda it, r2, omega: 1 / (np.sqrt(it))

FOLDER = "pictures/TCL/muL={0}".format(min(EIGENVALUES)/max(EIGENVALUES))
create_folder_if_not_existing(FOLDER)

COLORS = ["tab:blue", "tab:orange", "tab:brown", "tab:green", "tab:red", "tab:purple", "tab:cyan"]


def get_legend_limit_distrib():
    legend_color = [Line2D([0], [0], color=COLORS[0], lw=2, label='no compr.'),
                    Line2D([0], [0], color=COLORS[1], lw=2, label='compr.')]

    legend_line = [Line2D([0], [0], linestyle="--", color="black", lw=1, label=r"$\mathrm{Cov}( \sqrt{K} (\overline{w}_K - w_*))$"),
                   Line2D([0], [0], linestyle="-", color="black", lw=1,
                          label=r"$H^{-1} \mathfrak{C} H^{-1}$"),
                   Line2D([0], [0], linestyle=":", color="black", lw=1, label=r"$H^{-1} C H^{-1}$"),
                   Line2D([], [], color='black', marker="*", linestyle='None',
                          markersize=4, label=r"last $\bar w - w_*$")
                   ]
    return legend_line, legend_color


def plot_TCL_of_a_compressor(ax, sigma, lower_sigma, nb_clients, empirical_cov, avg_dist_to_opt, title):

    inv_sigma = compute_inversion(sigma)

    avg_dist_to_opt *= np.sqrt(SIZE_DATASET)

    ax_max = plot_ellipse(compute_empirical_covariance(avg_dist_to_opt), r"$\mathrm{Cov}( \sqrt{K} (\overline{w}_K - w_*))$", ax,
                       color=COLORS[1], zorder=0, lw=LINESIZE)
    plot_ellipse(compute_limit_distrib(inv_sigma, empirical_cov),
                       r"$H^{-1} \mathfrak{C}_{\mathrm{emp.}} H^{-1}$", ax, color=COLORS[1],
                       linestyle="--", marker="x", markevery=100, zorder=0, lw=LINESIZE)
    plot_ellipse(lower_sigma * inv_sigma/nb_clients,  r"$\sigma^2 H^{-1}/N$", ax,
                       color=COLORS[0], zorder=0, lw=LINESIZE, linestyle=":")

    ax.scatter(avg_dist_to_opt[:, 0], avg_dist_to_opt[:, 1], color=COLORS[1], alpha=0.5)

    ax.scatter(0, 0, c='red', s=3)
    ax.axvline(c='grey', lw=1)
    ax.axhline(c='grey', lw=1)
    ax_max *= 1.1
    ax.set_xlim(-ax_max, ax_max)
    ax.set_ylim(-ax_max, ax_max)
    ax.set_title(title, fontsize=FONTSIZE)


def plot_TCL(sigma, lower_sigma, nb_clients, all_covariances, all_avg_sgd, labels):
    fig, ax = plt.subplots(figsize=(6, 5))
    plot_TCL_of_a_compressor(ax, sigma, lower_sigma, nb_clients, all_covariances[0], all_avg_sgd[0], title=labels[0])
    ax.legend(loc='upper left', fancybox=True, framealpha=0.5)
    filename = "{0}/C{1}-N{2}-{3}-TCL_without_compression".format(FOLDER, NB_CLIENTS, SIZE_DATASET,HETEROGENEITY)
    if USE_ORTHO_MATRIX:
        filename = "{0}-ortho".format(filename)
    plt.savefig("{0}.pdf".format(filename), bbox_inches='tight', dpi=600)

    fig_TCL, axes_TCL = plt.subplots(2, 3, figsize=(10, 6))
    axes_TCL = axes_TCL.flat
    for idx_compressor in range(1, len(all_covariances)):
        plot_TCL_of_a_compressor(axes_TCL[idx_compressor - 1], sigma, lower_sigma, nb_clients, all_covariances[idx_compressor],
                                 all_avg_sgd[idx_compressor], title=labels[idx_compressor])
    axes_TCL[0].legend(loc='upper left', fancybox=True, framealpha=0.5)
    filename = "{0}/C{1}-N{2}-{3}-TCL_with_compression".format(FOLDER, NB_CLIENTS, SIZE_DATASET, HETEROGENEITY)
    if USE_ORTHO_MATRIX:
        filename = "{0}-ortho".format(filename)
    plt.savefig("{0}.pdf".format(filename), bbox_inches='tight', dpi=600)


def plot_theory_TCL(sigma, lower_sigma, nb_clients, all_covariances, labels):
    fig_TCL, axes_TCL = plt.subplots(2, 3, figsize=(10, 6))
    axes_TCL = axes_TCL.flat
    for idx_compressor in range(1, len(all_covariances)):
        ax = axes_TCL[idx_compressor-1]
        inv_sigma = compute_inversion(sigma)
        plot_ellipse(compute_limit_distrib(inv_sigma, all_covariances[idx_compressor]),
                     r"$H^{-1} \mathfrak{C}_{\mathrm{emp.}} H^{-1}$", ax, plot_eig=True,
                     color=COLORS[1], zorder=0, lw=LINESIZE)

        theoretical_cov = get_theoretical_cov(synthetic_dataset, nb_clients, labels[idx_compressor])
        if theoretical_cov is not None:
            plot_ellipse(compute_limit_distrib(inv_sigma, theoretical_cov),
                         r"$H^{-1} \mathfrak{C}_{\mathrm{th.}} H^{-1}$", ax, plot_eig=False,
                         color=COLORS[1], zorder=0, lw=LINESIZE, linestyle="--", marker="x", markevery=100)

        plot_ellipse(lower_sigma * inv_sigma / nb_clients, r"$\sigma^2 H^{-1} / N$", ax, plot_eig=True,
                           color=COLORS[0], zorder=0, lw=LINESIZE)

        ax.scatter(0, 0, c='red', s=3)
        ax.axvline(c='grey', lw=1)
        ax.axhline(c='grey', lw=1)
        ax.set_title(labels[idx_compressor], fontsize=FONTSIZE)
        ax.axis('equal')

    axes_TCL[0].legend(loc='upper left', fancybox=True, framealpha=0.5)
    filename = "{0}/C{1}-N{2}-TCL_theoretical".format(FOLDER, NB_CLIENTS, SIZE_DATASET)
    if USE_ORTHO_MATRIX:
        filename = "{0}-ortho".format(filename)
    plt.savefig("{0}.pdf".format(filename), bbox_inches='tight', dpi=600)


def compute_covariance_of_compressors(clients, compressor):
    empirical_cov = []
    X_compressed = clients[0].dataset.X.copy()
    for i in range(X_compressed.shape[0]):
        sum = 0
        for client in clients:
            sum += client.dataset.X[i] * client.dataset.epsilon[i]
        X_compressed[i] = compressor.compress(sum / len(clients))
    return compute_empirical_covariance(X_compressed)


def get_all_covariances_of_compressors(clients: List[Client], my_compressors):
    all_covariances = []
    for compressor in tqdm(my_compressors):
        cov_matrix = compute_covariance_of_compressors(clients, compressor)
        all_covariances.append(cov_matrix)

    return all_covariances


if __name__ == '__main__':

    clients = [Client(i, DIM, SIZE_DATASET // NB_CLIENTS, POWER_COV, NB_CLIENTS, USE_ORTHO_MATRIX, HETEROGENEITY,
                      eigenvalues=EIGENVALUES, w0_seed=None) for i in
               range(NB_CLIENTS)]
    check_clients(clients, HETEROGENEITY)
    synthetic_dataset = clients[0].dataset
    w_star = np.mean([client.dataset.w_star for client in clients], axis=0)

    no_compressor = Quantization(0, dim=DIM)
    my_compressors = [no_compressor, synthetic_dataset.quantizator, synthetic_dataset.stabilized_quantizator,
                      synthetic_dataset.sketcher, synthetic_dataset.sparsificator, synthetic_dataset.rand1,
                      synthetic_dataset.all_or_nothinger]
    labels = ["No compression"] + [compressor.get_name() for compressor in my_compressors[1:]]

    all_covariances = get_all_covariances_of_compressors(clients, my_compressors)

    plot_theory_TCL(synthetic_dataset.upper_sigma, len(clients), synthetic_dataset.lower_sigma, all_covariances, labels)

    all_avg_sgd = [[] for idx_compressor in range(len(my_compressors))]
    all_sgd_descent = []

    for idx in range(NB_TRY):

        print("TRY {0}/{1}".format(idx, NB_TRY))

        ### Very important to regenerate dataset in order to compute the variance of the last avg iterate. ###
        for client in clients:
            client.dataset.regenerate_dataset()

        for idx_compressor in range(len(my_compressors)):
            compressor = my_compressors[idx_compressor]
            sgd = SGDArtemis(clients, step_size, compressor, sto=True).gradient_descent(label=compressor.get_name())
            # We save only the excess loss of the first try.
            if idx == 0:
                all_sgd_descent.append(sgd)
            all_avg_sgd[idx_compressor].append([sgd.last_w - w_star])

        # We plot the excess loss of the first try.
        if idx == 0:
            sgd_series = SeriesOfSGD()
            sgd_series.append(all_sgd_descent)
            plot_only_avg(sgd_series, optimal_loss=0,
                          hash_string="TCL/muL={0}/C{1}-N{2}-{3}-avg_sgd".format(min(EIGENVALUES)/max(EIGENVALUES),
                                                                             NB_CLIENTS, SIZE_DATASET, HETEROGENEITY))

    all_avg_sgd = [np.concatenate(avg_sgd) for avg_sgd in all_avg_sgd]
    plot_TCL(synthetic_dataset.upper_sigma, synthetic_dataset.lower_sigma, len(clients), all_covariances, all_avg_sgd, labels)


