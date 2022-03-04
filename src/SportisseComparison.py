"""
Created by Constantin Philippenko, 3th March 2022.
"""
import copy
import math
import sys

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal
from scipy.stats import ortho_group
from tqdm import tqdm

from src.CompressionModel import RandomSparsification, SQuantization
from src.SyntheticDataset import SyntheticDataset, MAX_SIZE_DATASET
from src.Utilities import create_folder_if_not_existing

MAX_LOSS = 10**4

DISABLE = True

SIZE_DATASET = 10**5
DIM = 400
POWER_COV = 4
R_SIGMA=0

USE_ORTHO_MATRIX = False


def log_sampling_xaxix(size_dataset):
    log_len = np.int(math.log10(size_dataset))
    residual_len = math.log10(size_dataset) - log_len
    log_xaxis = [[math.pow(10, a) * math.pow(10, i / 100) for i in range(100)] for a in range(log_len)]
    log_xaxis.append([math.pow(10, log_len) * math.pow(10, i / 100) for i in range(int(100 * residual_len))])
    log_xaxis = np.concatenate(log_xaxis, axis=None)
    log_xaxis = np.unique(log_xaxis.astype(int))
    return log_xaxis

LOG_XAXIS = log_sampling_xaxix(SIZE_DATASET)

dataset = SyntheticDataset()


def compute_empirical_risk(w, true_risk: bool = True):
    if true_risk:
        diff_w = (w - dataset.w_star)
        return 0.5 * diff_w.T @ dataset.upper_sigma @ diff_w
    return 0.5 * np.linalg.norm(dataset.X_complete @ w - dataset.Y) ** 2 / len(dataset.Y)


def compute_NA_sportisse_gradient(w, index, p):
    x, y = dataset.X[index % MAX_SIZE_DATASET], dataset.Y[index % MAX_SIZE_DATASET]
    g = x * (w @ x - y) - (1 - p) * np.diag(x**2) @ w
    return g


def compute_stochastic_gradient(w, index):
    x, y = dataset.X_complete[index % MAX_SIZE_DATASET], dataset.Y[index % MAX_SIZE_DATASET]
    g = x * (w @ x - y)
    return g


def compute_sportisse_gradient(w, index, p):
    x, y = dataset.X_complete[index % MAX_SIZE_DATASET], dataset.Y[index % MAX_SIZE_DATASET]
    x = x * dataset.D[index % MAX_SIZE_DATASET] / p
    g = x * (w @ x - y) - (1 - p) * np.diag(x**2) @ w
    return g


def sportisse_NA():
    it = 1
    current_w = dataset.w0
    avg_w = copy.deepcopy(current_w)
    losses = [compute_empirical_risk(current_w)]
    avg_losses = [losses[-1]]
    for idx in tqdm(np.arange(dataset.size_dataset), disable=DISABLE):
        if idx % MAX_SIZE_DATASET == 0 and idx != 0:
            print("Regenerating ...")
            dataset.regenerate_dataset()
        it += 1
        grad = compute_NA_sportisse_gradient(current_w, idx, dataset.estimated_p)
        current_w = current_w - dataset.gamma * grad
        avg_w = current_w / it + avg_w * (it - 1) / it
        if idx in LOG_XAXIS[1:]:
            losses.append(compute_empirical_risk(current_w))
            avg_losses.append(compute_empirical_risk(avg_w))
        if losses[-1] == math.inf or losses[-1] > 1e9:
            losses[-1] = MAX_LOSS
            losses = losses + [losses[-1] for i in range(len(LOG_XAXIS) - len(losses))]
            avg_losses = avg_losses + [avg_losses[-1] for i in range(len(LOG_XAXIS) - len(avg_losses))]
            break
    return losses, avg_losses


def sportisse_Rdk():
    it = 1
    current_w = dataset.w0
    avg_w = copy.deepcopy(current_w)
    losses = [compute_empirical_risk(current_w)]
    avg_losses = [losses[-1]]
    for idx in tqdm(np.arange(dataset.size_dataset), disable=DISABLE):
        if idx % MAX_SIZE_DATASET == 0 and idx != 0:
            print("Regenerating ...")
            dataset.regenerate_dataset()
        it += 1
        grad = compute_sportisse_gradient(current_w, idx, dataset.estimated_p)
        current_w = current_w - dataset.gamma * grad
        avg_w = current_w / it + avg_w * (it - 1) / it
        if idx in LOG_XAXIS[1:]:
            losses.append(compute_empirical_risk(current_w))
            avg_losses.append(compute_empirical_risk(avg_w))
        if losses[-1] == math.inf or losses[-1] > 1e9:
            losses[-1] = MAX_LOSS
            losses = losses + [losses[-1] for i in range(len(LOG_XAXIS) - len(losses))]
            avg_losses = avg_losses + [avg_losses[-1] for i in range(len(LOG_XAXIS) - len(avg_losses))]
            break
    return losses, avg_losses


def compression_SGD(compressor):
    it = 1
    current_w = dataset.w0
    avg_w = copy.deepcopy(current_w)
    losses = [compute_empirical_risk(current_w)]
    avg_losses = [losses[-1]]
    for idx in tqdm(np.arange(dataset.size_dataset), disable=DISABLE):
        if idx % MAX_SIZE_DATASET == 0 and idx != 0:
            print("Regenerating ...")
            dataset.regenerate_dataset()
        it += 1
        grad = compute_stochastic_gradient(current_w, idx)
        grad = compressor.compress(grad)
        current_w = current_w - dataset.gamma * grad
        avg_w = current_w / it + avg_w * (it - 1) / it
        if idx in LOG_XAXIS[1:]:
            losses.append(compute_empirical_risk(current_w))
            avg_losses.append(compute_empirical_risk(avg_w))

        if losses[-1] == math.inf or losses[-1] > 1e9:
            losses[-1] = MAX_LOSS
            losses = losses + [losses[-1] for i in range(len(LOG_XAXIS) - len(losses))]
            avg_losses = avg_losses + [avg_losses[-1] for i in range(len(LOG_XAXIS) - len(avg_losses))]
            break
    return losses, avg_losses


def setup_plot_with_SGD(losses, avg_losses, labels, optimal_loss, picture_name: str):
    fig, axes = plt.subplots(2, figsize=(8, 7))

    for (l, avg_l, label) in zip(losses, avg_losses, labels):
        axes[0].plot(np.log10(LOG_XAXIS), np.log10(l - optimal_loss), label="SGD {0}".format(label), linewidth=3)
        axes[1].plot(np.log10(LOG_XAXIS), np.log10(avg_l - optimal_loss), label="AvgSGD {0}".format(label), linewidth=3)

    for ax in axes:
        ax.legend(loc='lower left', fontsize=15)
        ax.grid(True)
        ax.set_ylim(top=1)
    axes[0].set_ylabel(r"$\log_{10}(F(w_k) - F(w_*))$", fontsize=15)
    axes[1].set_ylabel(r"$\log_{10}(F(\bar w_k) - F(w_*))$", fontsize=15)
    axes[1].set_xlabel(r"$\log_{10}(n)$", fontsize=15)
    # plt.show()
    plt.savefig(picture_name + ".eps", format='eps')

def run(dim, power_cov, gamma, use_ortho_matrix):

    dataset.generate_dataset(dim, size_dataset=SIZE_DATASET, power_cov=power_cov, r_sigma=R_SIGMA,
                             use_ortho_matrix=use_ortho_matrix, do_logistic_regression=False)
    dataset.gamma = gamma
    folder = "./pictures/sportisse/"
    create_folder_if_not_existing(folder)
    picture_name = folder + "N{0}-D{1}-P{2}-g{3}-p{4:1.2f}".format(SIZE_DATASET, dim, power_cov, gamma, dataset.LEVEL_RDK)
    if use_ortho_matrix:
        picture_name = picture_name + "-ortho"

    SEED = 25

    all_losses = []
    all_avg_losses, all_labels = [], []

    np.random.seed(SEED)
    losses_vanilla, avg_losses_vanilla = compression_SGD(SQuantization(0, DIM))
    all_losses.append(losses_vanilla)
    all_avg_losses.append(avg_losses_vanilla)
    all_labels.append("Vanilla")

    # np.random.seed(SEED)
    # losses_NA, avg_losses_NA = sportisse_NA()
    # all_losses.append(losses_NA)
    # all_avg_losses.append(avg_losses_NA)
    # all_labels.append("SportisseNA")

    np.random.seed(SEED)
    losses_sportisseRdk, avg_losses_sportisseRdk = sportisse_Rdk()
    all_losses.append(losses_sportisseRdk)
    all_avg_losses.append(avg_losses_sportisseRdk)
    all_labels.append("SportisseRdk")

    np.random.seed(SEED)
    losses_Rdk, avg_losses_Rdk = compression_SGD(dataset.sparsificator)
    all_losses.append(losses_Rdk)
    all_avg_losses.append(avg_losses_Rdk)
    all_labels.append("Rdk")

    np.random.seed(SEED)
    losses_Qtz, avg_losses_Qtz = compression_SGD(dataset.quantizator)
    all_losses.append(losses_Qtz)
    all_avg_losses.append(avg_losses_Qtz)
    all_labels.append("Qtz")

    optimal_loss = compute_empirical_risk(dataset.w_star)

    setup_plot_with_SGD(all_losses, all_avg_losses, all_labels, optimal_loss, picture_name)


if __name__ == '__main__':
    for power_cov in [1,2,3,4,5,6]:
        for gamma in [0.1, 0.01, 0.001]:
                run(dim=int(sys.argv[1]), power_cov=power_cov, gamma=gamma, use_ortho_matrix=True)



