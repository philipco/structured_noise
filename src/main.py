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

from src.SGD import SGDRun, SeriesOfSGD, SGDVanilla, SGDCompressed
from src.SyntheticDataset import SyntheticDataset

SIZE_DATASET = 10**7
DIM = 100
POWER_COV = 4
R_SIGMA=0

USE_ORTHO_MATRIX = True
DO_LOGISTIC_REGRESSION = False


def plot_SGD_and_AVG(axes, sgd_run: SGDRun, optimal_loss):
    losses, avg_losses = np.array(sgd_run.losses), np.array(sgd_run.avg_losses)

    # Uniform sampling of a log-xaxis
    log_len = np.int(math.log10(len(losses)))
    residual_len = math.log10(len((losses))) - log_len
    xaxis = [[math.pow(10, a) * math.pow(10, i/100) for i in range(100)] for a in range(log_len)]
    xaxis.append([math.pow(10, log_len) * math.pow(10, i/100) for i in range(int(100 * residual_len))])
    xaxis = np.concatenate(xaxis, axis=None)
    xaxis = np.unique(xaxis.astype(int))

    axes[0].plot(np.log10(xaxis), np.log10(np.take(losses, xaxis) - optimal_loss), label="SGD {0}".format(sgd_run.label))
    axes[1].plot(np.log10(xaxis), np.log10(np.take(avg_losses, xaxis) - optimal_loss), label="AvgSGD {0}".format(sgd_run.label))


def setup_plot(losses1, avg_losses1, label1, losses2, avg_losses2, label2, optimal_loss):
    fig, ax = plt.subplots(figsize=(8, 7))
    plot_SGD_and_AVG(losses1, avg_losses1, optimal_loss, label1)
    plot_SGD_and_AVG(losses2, avg_losses2, optimal_loss, label2)

    # ax.set_yscale('log')
    ax.label(loc='best', fontsize=15)
    ax.set_xlabel(r"$\log_{10}(n)$", fontsize=15)
    ax.set_ylabel(r"$\log_{10}(F(w_k) - F(w_*))$", fontsize=15)
    ax.grid(True)
    # ax.set_ylim(top=10)
    plt.show()


def setup_plot_with_SGD(sgd_nocompr: SGDRun, sgd_try1: SGDRun, sgd_try2: SGDRun, optimal_loss, hash_string:str=None):
    fig, axes = plt.subplots(2, figsize=(8, 7))
    plot_SGD_and_AVG(axes, sgd_nocompr, optimal_loss)
    plot_SGD_and_AVG(axes, sgd_try1, optimal_loss)
    plot_SGD_and_AVG(axes, sgd_try2, optimal_loss)

    for ax in axes:
        ax.legend(loc='best', fontsize=15)
        ax.set_ylabel(r"$\log_{10}(F(w_k) - F(w_*))$", fontsize=15)
        ax.grid(True)
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

    vanilla_sgd = SGDVanilla(copy.deepcopy(synthetic_dataset))
    sgd_nocompr = vanilla_sgd.gradient_descent(label="no compression")
    optimal_loss = vanilla_sgd.compute_true_risk(synthetic_dataset.w_star, None, None)

    # losses_noised, avg_losses_noised, w = sgd.gradient_descent_noised()
    # setup_plot(losses, avg_losses, "", losses_noised, avg_losses_noised, "noised", optimal_loss)

    sgd_qtz = SGDCompressed(copy.deepcopy(synthetic_dataset), synthetic_dataset.quantizator).gradient_descent(label="quantization")
    sgd_rdk = SGDCompressed(copy.deepcopy(synthetic_dataset), synthetic_dataset.sparsificator).gradient_descent(label="sparsification")

    sgd_series = SeriesOfSGD(sgd_nocompr, sgd_qtz, sgd_rdk)
    sgd_series.save("pickle/" + synthetic_dataset.string_for_hash())

    setup_plot_with_SGD(sgd_nocompr, sgd_qtz, sgd_rdk, 0, hash_string=synthetic_dataset.string_for_hash())


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

    # fig, ax = plt.subplots(figsize=(8, 7))
    # plt.plot(np.log10(np.arange(1, DIM + 1)), np.log10(sgd_nocompr.diag_cov_gradients), label="No compression")
    # plt.plot(np.log10(np.arange(1, DIM + 1)), np.log10(sgd_qtz.diag_cov_gradients), label="Quantization")
    # plt.plot(np.log10(np.arange(1, DIM + 1)), np.log10(sgd_rdk.diag_cov_gradients), label="Sparsification")
    # ax.tick_params(axis='both', labelsize=15)
    # ax.legend(loc='best', fontsize=15)
    # ax.set_xlabel(r"$\log(i), \forall i \in \{1, ..., d\}$", fontsize=15)
    # ax.set_ylabel(r"$\log(Diag(\frac{X^T.X}{n})_i)$", fontsize=15)
    # plt.legend(loc='best', fontsize=15)
    # plt.show()
