"""Created by Constantin Philippenko, 11th April 2022."""

# axmax = max ellipse ! Sinon, outliners !
from typing import List

import sympy as sy

from src.federated_learning.Client import Client, check_clients

sy.init_printing(use_unicode=True)

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm

from src.PlotUtils import confidence_ellipse, plot_ellipse
from src.CompressionModel import *
from src.SGD import SGDCompressed
from src.SyntheticDataset import SyntheticDataset
from src.TheoreticalCov import get_theoretical_cov
from src.Utilities import create_folder_if_not_existing
from src.main import plot_only_avg

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

# TOOOODOOOOO : plt.axis('equal') pour que les axes aient la même taile !!!!!

SIZE_DATASET = 5*10**4
POWER_COV = 4
R_SIGMA = 0

DIM = 2
EIGENVALUES = np.array([1,10])


FONTSIZE = 17
LINESIZE = 3

NB_CLIENTS = 1
HETEROGENEITY = "homog" # "wstar" "sigma" "homog"

USE_ORTHO_MATRIX = True
DO_LOGISTIC_REGRESSION = False

LAST_POINTS = 0
NB_TRY = 20

FOLDER = "pictures/TCL/muL={0}".format(min(EIGENVALUES)/max(EIGENVALUES))
create_folder_if_not_existing(FOLDER)

COLORS = ["tab:blue", "tab:orange", "tab:brown", "tab:green", "tab:red", "tab:purple", "tab:cyan"]


def compute_inversion(matrix):
    return np.linalg.inv(matrix)


def compute_limit_distrib(inv_sigma, error_cov):
    return inv_sigma @ error_cov @ inv_sigma


def compute_empirical_covariance(random_vector):
    return random_vector.T.dot(random_vector) / random_vector.shape[0]


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


def plot_TCL_of_a_compressor(ax, sigma, empirical_cov, avg_dist_to_opt, title):

    inv_sigma = compute_inversion(sigma)

    avg_dist_to_opt *= np.sqrt(SIZE_DATASET)

    ax_max = plot_ellipse(compute_empirical_covariance(avg_dist_to_opt), r"$\mathrm{Cov}( \sqrt{n} (\bar w_n - w_*))$", ax,
                       color=COLORS[1], zorder=0, lw=LINESIZE)
    plot_ellipse(compute_limit_distrib(inv_sigma, empirical_cov),
                       r"$\Sigma^{-1} \mathrm{Cov}_{\mathrm{emp.}}(\mathcal C (x)) \Sigma^{-1}$", ax, color=COLORS[1],
                       linestyle="--", zorder=0, lw=LINESIZE)
    plot_ellipse(inv_sigma,  r"$\Sigma^{-1}$", ax,
                       color=COLORS[0], zorder=0, lw=LINESIZE, linestyle=":")

    ax.scatter(avg_dist_to_opt[:, 0], avg_dist_to_opt[:, 1], color=COLORS[1], alpha=0.5)

    ax.scatter(0, 0, c='red', s=3)
    ax.axvline(c='grey', lw=1)
    ax.axhline(c='grey', lw=1)
    ax_max *= 1.1
    ax.set_xlim(-ax_max, ax_max)
    ax.set_ylim(-ax_max, ax_max)
    ax.set_title(title, fontsize=FONTSIZE)


def plot_TCL(sigma, all_covariances, all_avg_sgd, labels):
    fig, ax = plt.subplots(figsize=(6, 5))
    plot_TCL_of_a_compressor(ax, sigma, all_covariances[0], all_avg_sgd[0], title=labels[0])
    ax.legend(loc='upper right', fancybox=True, framealpha=0.5)
    filename = "{0}/C{1}-N{2}-TCL_without_compression".format(FOLDER, NB_CLIENTS, SIZE_DATASET)
    if USE_ORTHO_MATRIX:
        filename = "{0}-ortho".format(filename)
    plt.savefig("{0}.png".format(filename), bbox_inches='tight', dpi=600)

    fig_TCL, axes_TCL = plt.subplots(2, 3, figsize=(10, 6))
    axes_TCL = axes_TCL.flat
    for idx_compressor in range(1, len(all_covariances)):
        plot_TCL_of_a_compressor(axes_TCL[idx_compressor - 1], sigma, all_covariances[idx_compressor],
                                 all_avg_sgd[idx_compressor], title=labels[idx_compressor])
    axes_TCL[0].legend(loc='upper right', fancybox=True, framealpha=0.5)
    filename = "{0}/C{1}-N{2}-TCL_with_compression".format(FOLDER, NB_CLIENTS, SIZE_DATASET)
    if USE_ORTHO_MATRIX:
        filename = "{0}-ortho".format(filename)
    plt.savefig("{0}.png".format(filename), bbox_inches='tight', dpi=600)


def plot_theory_TCL(sigma, all_covariances, labels):
    fig_TCL, axes_TCL = plt.subplots(2, 3, figsize=(10, 6))
    axes_TCL = axes_TCL.flat
    for idx_compressor in range(1, len(all_covariances)):
        ax = axes_TCL[idx_compressor-1]
        inv_sigma = compute_inversion(sigma)
        plot_ellipse(compute_limit_distrib(inv_sigma, all_covariances[idx_compressor]),
                     r"$\Sigma^{-1} \mathrm{Cov}_{\mathrm{emp.}}(\mathcal C (x)) \Sigma^{-1}$", ax, plot_eig=True,
                     color=COLORS[1], zorder=0, lw=LINESIZE)

        theoretical_cov = get_theoretical_cov(synthetic_dataset, labels[idx_compressor])
        if theoretical_cov is not None:
            plot_ellipse(compute_limit_distrib(inv_sigma, theoretical_cov),
                         r"$\Sigma^{-1} \mathrm{Cov}_{\mathrm{th.}}(\mathcal C (x)) \Sigma^{-1}$", ax, plot_eig=False,
                         color=COLORS[1], zorder=0, lw=LINESIZE, linestyle="--", marker="x",markevery=100)

        plot_ellipse(inv_sigma, r"$\Sigma^{-1}$", ax, plot_eig=True,
                           color=COLORS[0], zorder=0, lw=LINESIZE)

        ax.scatter(0, 0, c='red', s=3)
        ax.axvline(c='grey', lw=1)
        ax.axhline(c='grey', lw=1)
        ax.set_title(labels[idx_compressor], fontsize=FONTSIZE)
        ax.axis('equal')

    axes_TCL[0].legend(loc='upper right', fancybox=True, framealpha=0.5)
    filename = "{0}/C{1}-N{2}-TCL_theoretical".format(FOLDER, NB_CLIENTS, SIZE_DATASET)
    if USE_ORTHO_MATRIX:
        filename = "{0}-ortho".format(filename)
    plt.savefig("{0}.png".format(filename), bbox_inches='tight', dpi=600)


def compute_covariance_of_compressors(dataset, compressor):
    empirical_cov = []
    for client in clients:
        X = client.dataset.X_complete
        X_compressed = X.copy()
        for i in range(X.shape[0]):
            X_compressed[i] = compressor.compress(X[i])
    empirical_cov.append(compute_empirical_covariance(X_compressed))
    return np.mean(empirical_cov, axis = 0)


def get_all_covariances_of_compressors(clients: List[Client], my_compressors):
    all_covariances = []
    for compressor in tqdm(my_compressors):
        cov_matrix = compute_covariance_of_compressors(clients, compressor)
        all_covariances.append(cov_matrix)

    return all_covariances


def compute_theory_TCL():
    A = EIGENVALUES[0]
    B = EIGENVALUES[1]
    a, b, theta = sy.symbols('a b theta')
    D = sy.Matrix([[a, 0], [0, b]])
    Q = sy.Matrix([[sy.cos(theta), -sy.sin(theta)], [sy.sin(theta), sy.cos(theta)]])
    Sigma = Q @ D @ Q.T
    inv_Sigma = Sigma ** -1

    # Create covariance matrix for sparsification
    p = sy.symbols('p')
    P = sy.Matrix([[1 / p, 1], [1, 1 / p]])
    C_s = Sigma.multiply_elementwise(P)

    # Create covariance matrix for quantization
    diagonalizer = sy.Matrix([[1, 0], [0, 1]])
    C_q = Sigma + sy.sqrt(sy.Trace(Sigma)) * sy.sqrt(
        Sigma.multiply_elementwise(diagonalizer)) - Sigma.multiply_elementwise(diagonalizer)

    TCL_q = inv_Sigma @ C_q @ inv_Sigma
    TCL_s = inv_Sigma @ C_s @ inv_Sigma

    print("TCL covariances:")
    print("Inverse of sigma")
    print(inv_Sigma.subs([(theta, sy.pi / 4)]))
    print(inv_Sigma.subs([(a, A), (b, B), (theta, sy.pi / 4)]))
    print(inv_Sigma.subs([(a, A), (b, B)])) # Theta is still formal.
    print("Quantization")
    print(TCL_q.subs([(a, A), (b, B), (theta, sy.pi / 4)]))
    print(TCL_q.subs([(a, A), (b, B)])) # Theta is still formal.
    print("Sparsification")
    print(TCL_s.subs([(a, A), (b, B), (theta, sy.pi / 4)]))
    print(TCL_s.subs([(a, A), (b, B), (theta, sy.pi / 4), (p, 0.72)]))

    print("Eigenvalues:")
    inv_Sigma_eigenvalues = inv_Sigma.eigenvects()
    print("Inverse of sigma")
    for eig in inv_Sigma_eigenvalues:
        print(eig[2][0].subs([(a, 1), (b, 10), (theta, sy.pi / 4)]))
    # TCL_eigenvalues_q = TCL_q.eigenvects()
    # print("Quantization")
    # for eig in TCL_eigenvalues_q:
    #     print(eig[2][0].subs([(a, 1), (b, 10), (theta, sy.pi / 4)]))
    # TCL_eigenvalues_s = TCL_s.eigenvects()
    # print("Sparsification")
    # for eig in TCL_eigenvalues_s:
    #     print(eig[2][0].subs([(a, 1), (b, 10), (theta, sy.pi / 4), (p, 0.72)]))


if __name__ == '__main__':

    clients = [Client(DIM, SIZE_DATASET // NB_CLIENTS, POWER_COV, NB_CLIENTS, USE_ORTHO_MATRIX, HETEROGENEITY) for i in
               range(NB_CLIENTS)]
    check_clients(clients, HETEROGENEITY)
    synthetic_dataset = clients[0].dataset

    no_compressor = SQuantization(0, dim=DIM)
    my_compressors = [no_compressor, synthetic_dataset.quantizator, synthetic_dataset.stabilized_quantizator,
                      synthetic_dataset.rand_sketcher,
                      synthetic_dataset.sparsificator, synthetic_dataset.rand1,
                      synthetic_dataset.all_or_nothinger]
    labels = ["No compression"] + [compressor.get_name() for compressor in my_compressors[1:]]

    all_covariances = get_all_covariances_of_compressors(clients, my_compressors)

    plot_theory_TCL(synthetic_dataset.upper_sigma, all_covariances, labels)
    # compute_theory_TCL()

    all_avg_sgd = [[] for idx_compressor in range(len(my_compressors))]
    all_sgd_descent = []

    for idx in range(NB_TRY):

        ### Very important to regenerate dataset in order to compute the variance of the last avg iterate. ###
        for client in clients:
            client.dataset.regenerate_dataset()

        for idx_compressor in range(len(my_compressors)):
            compressor = my_compressors[idx_compressor]
            sgd = SGDCompressed(clients, compressor).gradient_descent(label=compressor.get_name(),
                                                                                deacreasing_step_size=True)
            # We save only the excess loss of the first try.
            if idx == 0:
                all_sgd_descent.append(sgd)
            all_avg_sgd[idx_compressor].append(sgd.all_avg_w[-1:] - synthetic_dataset.w_star)

        # We plot the excess loss of the first try.
        if idx == 0:
            plot_only_avg(all_sgd_descent[1:], sgd_nocompr=all_sgd_descent[0], optimal_loss=0,
                          hash_string="TCL/muL={0}/C{1}-N{2}-avg_sgd".format(min(EIGENVALUES)/max(EIGENVALUES),
                                                                             NB_CLIENTS, SIZE_DATASET))

    all_avg_sgd = [np.concatenate(avg_sgd) for avg_sgd in all_avg_sgd]
    plot_TCL(synthetic_dataset.upper_sigma, all_covariances, all_avg_sgd, labels)


