"""
Created by Constantin Philippenko, 20th December 2021.
"""
import copy
import hashlib
import math
import random

import numpy as np

import matplotlib
# matplotlib.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'text.usetex': True,
#     'pgf.rcfonts': False,
#     'text.latex.preamble': r'\usepackage{amsfonts}'
# })

from matplotlib import pyplot as plt

from src.SGD import SGDRun, SeriesOfSGD, SGDVanilla, SGDCompressed, SGDSparsification, SGDSportisse
from src.SyntheticDataset import SyntheticDataset

SIZE_DATASET = 10**5
DIM = 100
POWER_COV = 3
R_SIGMA=0

USE_ORTHO_MATRIX = True
DO_LOGISTIC_REGRESSION = False


def plot_SGD_and_AVG(axes, sgd_run: SGDRun, optimal_loss):

    axes[0].plot(np.log10(sgd_run.log_xaxis), np.log10(sgd_run.losses - optimal_loss),
                 label="SGD {0}".format(sgd_run.label))
    axes[1].plot(np.log10(sgd_run.log_xaxis), np.log10(sgd_run.avg_losses - optimal_loss),
                 label="AvgSGD {0}".format(sgd_run.label))


def setup_plot(losses1, avg_losses1, label1, losses2, avg_losses2, label2, optimal_loss):
    fig, ax = plt.subplots(figsize=(8, 7))
    plot_SGD_and_AVG(losses1, avg_losses1, optimal_loss, label1)
    plot_SGD_and_AVG(losses2, avg_losses2, optimal_loss, label2)

    # ax.set_yscale('log')
    ax.label(loc='best', fontsize=15)
    ax.set_xlabel(r"$\log_{10}(n)$", fontsize=15)
    ax[0].set_ylabel(r"$\log_{10}(F(w_k) - F(w_*))$", fontsize=15)
    ax[1].set_ylabel(r"$\log_{10}(F(\bar w_k) - F(w_*))$", fontsize=15)
    ax.grid(True)
    # ax.set_ylim(top=10)
    plt.show()


def setup_plot_with_SGD(*args, sgd_nocompr: SGDRun, optimal_loss, hash_string:str=None):
    fig, axes = plt.subplots(2, figsize=(8, 7))

    plot_SGD_and_AVG(axes, sgd_nocompr, optimal_loss)

    for sgd_try in args:
        plot_SGD_and_AVG(axes, sgd_try, optimal_loss)

    for ax in axes:
        ax.legend(loc='best', fontsize=15)
        ax.grid(True)
    axes[0].set_ylabel(r"$\log_{10}(F(w_k) - F(w_*))$", fontsize=15)
    axes[1].set_ylabel(r"$\log_{10}(F(\bar w_k) - F(w_*))$", fontsize=15)
    axes[1].set_xlabel(r"$\log_{10}(n)$", fontsize=15)

    if hash_string:
        plt.savefig('{0}.eps'.format("./pictures/" + hash_string), format='eps')
        plt.close()
    else:
        plt.show()

if __name__ == '__main__':

    synthetic_dataset = SyntheticDataset()
    synthetic_dataset.generate_dataset(DIM, size_dataset=SIZE_DATASET, power_cov=POWER_COV, r_sigma=R_SIGMA,
                                       use_ortho_matrix=USE_ORTHO_MATRIX, do_logistic_regression=DO_LOGISTIC_REGRESSION)

    hash_string = hashlib.shake_256(synthetic_dataset.string_for_hash().encode()).hexdigest(4)

    sgd_sportisse = SGDSportisse(copy.deepcopy(synthetic_dataset),
                                 synthetic_dataset.sparsificator).gradient_descent(label="sportisse")

    vanilla_sgd = SGDVanilla(copy.deepcopy(synthetic_dataset))
    sgd_nocompr = vanilla_sgd.gradient_descent(label="no compression")

    w_ERM = (np.linalg.pinv(synthetic_dataset.X_complete.T.dot(synthetic_dataset.X_complete))
             .dot(synthetic_dataset.X_complete.T)).dot(synthetic_dataset.Y)
    optimal_loss = vanilla_sgd.compute_true_risk(w_ERM, synthetic_dataset.X_complete,
                                                      synthetic_dataset.Y)

    # losses_noised, avg_losses_noised, w = sgd.gradient_descent_noised()
    # setup_plot(losses, avg_losses, "", losses_noised, avg_losses_noised, "noised", optimal_loss)

    sgd_rdk = SGDSparsification(copy.deepcopy(synthetic_dataset),
                                synthetic_dataset.sparsificator).gradient_descent(label="sparsification")
    sgd_qtz = SGDCompressed(copy.deepcopy(synthetic_dataset), synthetic_dataset.quantizator).gradient_descent(
        label="quantization")

    sgd_series = SeriesOfSGD(sgd_nocompr, sgd_rdk)
    sgd_series.save("pickle/" + synthetic_dataset.string_for_hash())

    setup_plot_with_SGD(sgd_rdk, sgd_sportisse, sgd_nocompr=sgd_nocompr, optimal_loss=optimal_loss,
                        hash_string=synthetic_dataset.string_for_hash())

    setup_plot_with_SGD(sgd_rdk, sgd_sportisse, sgd_qtz, sgd_nocompr=sgd_nocompr, optimal_loss=optimal_loss,
                        hash_string=synthetic_dataset.string_for_hash())


    # plt.imshow(matrix_grad)
    # plt.colorbar()
    # plt.title("No compression", fontsize=15)
    # plt.show()
    #
    # plt.imshow(matrix_qtz)
    # plt.colorbar()
    # plt.title("Quantization", fontsize=15)
    # plt.show()
    #
    # plt.imshow(matrix_sparse)
    # plt.colorbar()
    # plt.title("Sparsification", fontsize=15)
    # plt.show()

    fig, ax = plt.subplots(figsize=(8, 7))
    plt.plot(np.log10(np.arange(1, DIM + 1)), np.log10(sgd_nocompr.diag_cov_gradients), label=sgd_nocompr.label)
    plt.plot(np.log10(np.arange(1, DIM + 1)), np.log10(sgd_rdk.diag_cov_gradients), label=sgd_rdk.label)
    plt.plot(np.log10(np.arange(1, DIM + 1)), np.log10(sgd_sportisse.diag_cov_gradients), label=sgd_sportisse.label)
    plt.plot(np.log10(np.arange(1, DIM + 1)), np.log10(sgd_qtz.diag_cov_gradients), label=sgd_qtz.label)
    ax.tick_params(axis='both', labelsize=15)
    ax.legend(loc='best', fontsize=15)
    ax.set_xlabel(r"$\log(i), \forall i \in \{1, ..., d\}$", fontsize=15)
    ax.set_ylabel(r"$\log(Diag(\frac{X^T.X}{n})_i)$", fontsize=15)
    plt.legend(loc='best', fontsize=15)
    plt.show()
