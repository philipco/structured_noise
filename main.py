"""
Created by Constantin Philippenko, 20th December 2021.
"""
import copy
import random

import numpy as np
from matplotlib import pyplot as plt
from numpy.random.mtrand import multivariate_normal

from tqdm import tqdm

from CompressionModel import CompressionModel, RandomSparsification, SQuantization, RandomDithering, SignSGD
from SGD import SGD
from SyntheticDataset import SyntheticDataset

SIZE_DATASET = 10000
DIM = 50


def plot_SGD_and_AVG(losses, avg_losses, optimal_loss, label):
    plt.plot(np.log10(np.arange(1, len(losses) + 1)), np.log10(losses - optimal_loss), label="SGD {0}".format(label))
    plt.plot(np.log10(np.arange(1, len(losses) + 1)), np.log10(avg_losses - optimal_loss), label="Avg SGD {0}".format(label), marker="v")


def setup_plot(losses1, avg_losses1, label1, losses2, avg_losses2, label2, optimal_loss):
    fig, ax = plt.subplots(figsize=(8, 7))
    plot_SGD_and_AVG(losses1, avg_losses1, optimal_loss, label1)
    plot_SGD_and_AVG(losses2, avg_losses2, optimal_loss, label2)

    # ax.set_yscale('log')
    ax.legend(loc='best', fontsize=15)
    ax.set_xlabel(r"$\log_{10}(n)$", fontsize=15)
    ax.set_ylabel(r"$\log_{10}(F(w_k) - F(w_*))$", fontsize=15)
    # ax.set_ylim(top=10)
    plt.show()


def setup_plot_with_SGD(loss_sgd, avg_loss_sgd, losses1, avg_losses1, label1, losses2, avg_losses2, label2, optimal_loss):
    fig, ax = plt.subplots(figsize=(8, 7))
    plot_SGD_and_AVG(loss_sgd, avg_loss_sgd, optimal_loss, "")
    plot_SGD_and_AVG(losses1, avg_losses1, optimal_loss, label1)
    plot_SGD_and_AVG(losses2, avg_losses2, optimal_loss, label2)

    # ax.set_yscale('log')
    ax.legend(loc='best', fontsize=15)
    ax.set_xlabel(r"$\log_{10}(n)$", fontsize=15)
    ax.set_ylabel(r"$\log_{10}(F(w_k) - F(w_*))$", fontsize=15)
    # ax.set_ylim(top=20)
    plt.show()

if __name__ == '__main__':

    synthetic_dataset = SyntheticDataset()
    synthetic_dataset.generate_dataset(DIM, size_dataset=SIZE_DATASET, power_cov=2, use_ortho_matrix=False)
    sgd = SGD(synthetic_dataset)
    optimal_loss = sgd.compute_true_risk(synthetic_dataset.w_star, synthetic_dataset.X, synthetic_dataset.Y)

    losses, avg_losses, w, matrix_grad = sgd.gradient_descent()

    # losses_noised, avg_losses_noised, w = sgd.gradient_descent_noised()
    # setup_plot(losses, avg_losses, "", losses_noised, avg_losses_noised, "noised", optimal_loss)

    losses_quantization, avg_losses_quantization, w, matrix_qtz = sgd.gradient_descent_compression(synthetic_dataset.quantizator)
    losses_random_sparsification, avg_losses_random_sparsification, w, matrix_sparse = \
        sgd.gradient_descent_compression(synthetic_dataset.sparsificator)
    setup_plot_with_SGD(losses, avg_losses, losses_quantization, avg_losses_quantization, "qtz", losses_random_sparsification,
               avg_losses_random_sparsification, "rdk", optimal_loss)

    p = synthetic_dataset.sparsificator.level
    P = p * np.identity(n=DIM)

    fig, ax = plt.subplots(figsize=(8, 7))
    plt.plot(np.log10(np.arange(1, DIM + 1)), np.log10(np.diag(matrix_grad)), label="No compression")
    plt.plot(np.log10(np.arange(1, DIM + 1)), np.log10(np.diag(matrix_qtz)), label="Quantization")
    plt.plot(np.log10(np.arange(1, DIM + 1)), np.log10(np.diag( matrix_sparse)), label="Sparsification")
    ax.tick_params(axis='both', labelsize=15)
    ax.legend(loc='best', fontsize=15)
    ax.set_xlabel(r"$\log(i), \forall i \in \{1, ..., d\}$", fontsize=15)
    ax.set_ylabel(r"$\log(Diag(\frac{X^T.X \Sigma}{n})_i)$", fontsize=15)
    plt.legend(loc='best', fontsize=15)
    plt.show()

    # range_dim =
    # expl = Exploration("UPPER SIGMA", range_hyperparameters, lambda_fn, metrics))


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
