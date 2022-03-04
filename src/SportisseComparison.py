"""
Created by Constantin Philippenko, 3th March 2022.
"""
import copy
import math

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal
from scipy.stats import ortho_group
from tqdm import tqdm

from src.CompressionModel import RandomSparsification, SQuantization

SIZE_DATASET = 10**6
DIM = 400
POWER_COV = 6
R_SIGMA=0
W_STAR = np.ones(DIM)
GAMMA = 0.01 # GAMMA 0.01

USE_ORTHO_MATRIX = False

LEVEL_QTZ = 1  # 1 # np.floor(np.sqrt(self.dim) / TARGET_OMEGA)  # Lead to omega_c = 3.
quantizator = SQuantization(LEVEL_QTZ, dim=DIM)

LEVEL_RDK = 1 / (quantizator.omega_c + 1)
sparsificator = RandomSparsification(LEVEL_RDK, dim=DIM, biased=False)

print("Level of sparsification:", LEVEL_RDK)

def log_sampling_xaxix(size_dataset):
    log_len = np.int(math.log10(size_dataset))
    residual_len = math.log10(size_dataset) - log_len
    log_xaxis = [[math.pow(10, a) * math.pow(10, i / 100) for i in range(100)] for a in range(log_len)]
    log_xaxis.append([math.pow(10, log_len) * math.pow(10, i / 100) for i in range(int(100 * residual_len))])
    log_xaxis = np.concatenate(log_xaxis, axis=None)
    log_xaxis = np.unique(log_xaxis.astype(int))
    return log_xaxis

LOG_XAXIS = log_sampling_xaxix(SIZE_DATASET)

upper_sigma = np.diag(np.array([1 / (i ** POWER_COV) for i in range(1, DIM + 1)]), k=0)
if USE_ORTHO_MATRIX:
    ortho_matrix = ortho_group.rvs(dim=DIM)
    upper_sigma = ortho_matrix @ upper_sigma @ ortho_matrix.T

np.random.seed(25)
X_complete = multivariate_normal(np.zeros(DIM), upper_sigma, size=SIZE_DATASET)
X = copy.deepcopy(X_complete)
D = copy.deepcopy(X_complete)

p = LEVEL_RDK
compress_data = RandomSparsification(p, dim=DIM, biased=False)
for i in range(SIZE_DATASET):
    D[i] = np.random.binomial(n=1, p=p, size=DIM)
    X[i] = X[i] * D[i] / p
estimated_p = 1 - np.count_nonzero(X == 0) / (SIZE_DATASET * DIM)
print("Estimated p:", estimated_p)

Y = X_complete @ W_STAR + np.random.normal(0, 1, size=SIZE_DATASET)
W_ERM = (np.linalg.pinv(X_complete.T.dot(X_complete)).dot(X_complete.T)).dot(Y)
W0 = np.random.normal(0, 1, size=DIM)


def compute_empirical_risk(w, true_risk: bool = True):
    if true_risk:
        diff_w = (w - W_STAR)
        return 0.5 * diff_w.T @ upper_sigma @ diff_w
    return 0.5 * np.linalg.norm(X_complete @ w - Y) ** 2 / len(Y)


def compute_NA_sportisse_gradient(w, index, p):
    x, y = X[index], Y[index]
    g = x * (w @ x - y) - (1 - p) * np.diag(x**2) @ w
    return g


def compute_stochastic_gradient(w, index):
    x, y = X_complete[index], Y[index]
    g = x * (w @ x - y)
    return g


def compute_sportisse_gradient(w, index, p):
    x, y = X_complete[index], Y[index]
    x = x * D[index] / p
    g = x * (w @ x - y) - (1 - p) * np.diag(x**2) @ w
    return g


def sportisse_NA():
    it = 1
    current_w = W0
    avg_w = copy.deepcopy(current_w)
    losses = [compute_empirical_risk(current_w)]
    avg_losses = [losses[-1]]
    for idx in tqdm(range(len(Y))):
        it += 1
        grad = compute_NA_sportisse_gradient(current_w, idx, estimated_p)
        current_w = current_w - GAMMA * grad
        avg_w = current_w / it + avg_w * (it - 1) / it
        if idx in LOG_XAXIS[1:]:
            losses.append(compute_empirical_risk(current_w))
            avg_losses.append(compute_empirical_risk(avg_w))
    return losses, avg_losses


def sportisse_Rdk():
    it = 1
    current_w = W0
    avg_w = copy.deepcopy(current_w)
    losses = [compute_empirical_risk(current_w)]
    avg_losses = [losses[-1]]
    for idx in tqdm(range(len(Y))):
        it += 1
        grad = compute_sportisse_gradient(current_w, idx, estimated_p)
        current_w = current_w - GAMMA * grad
        avg_w = current_w / it + avg_w * (it - 1) / it
        if idx in LOG_XAXIS[1:]:
            losses.append(compute_empirical_risk(current_w))
            avg_losses.append(compute_empirical_risk(avg_w))
    return losses, avg_losses


def compression_SGD(compressor):
    it = 1
    current_w = W0
    avg_w = copy.deepcopy(current_w)
    losses = [compute_empirical_risk(current_w)]
    avg_losses = [losses[-1]]
    for idx in tqdm(range(len(Y))):
        it += 1
        grad = compute_stochastic_gradient(current_w, idx)
        grad = compressor.compress(grad)
        current_w = current_w - GAMMA * grad
        avg_w = current_w / it + avg_w * (it - 1) / it
        if idx in LOG_XAXIS[1:]:
            losses.append(compute_empirical_risk(current_w))
            avg_losses.append(compute_empirical_risk(avg_w))
    return losses, avg_losses


def setup_plot_with_SGD(losses, avg_losses, labels, optimal_loss):
    fig, axes = plt.subplots(2, figsize=(8, 7))

    it = 0
    for (l, avg_l, label) in zip(losses, avg_losses, labels):
        axes[0].plot(np.log10(LOG_XAXIS), np.log10(l - optimal_loss), label="SGD {0}".format(label), linewidth=4)
        axes[1].plot(np.log10(LOG_XAXIS), np.log10(avg_l - optimal_loss), label="AvgSGD {0}".format(label), linewidth=4)

    for ax in axes:
        ax.legend(loc='best', fontsize=15)
        ax.grid(True)
    axes[0].set_ylabel(r"$\log_{10}(F(w_k) - F(w_*))$", fontsize=15)
    axes[1].set_ylabel(r"$\log_{10}(F(\bar w_k) - F(w_*))$", fontsize=15)
    axes[1].set_xlabel(r"$\log_{10}(n)$", fontsize=15)
    plt.show()
    hash = "pictures/sportisse/N{0}-D{1}-P{2}-p{3:1.2f}}.eps".format(SIZE_DATASET, DIM, POWER_COV, estimated_p)
    plt.savefig(hash, format='eps')

if __name__ == '__main__':
    SEED = 25

    all_losses = []
    all_avg_losses, all_labels = [], []

    # np.random.seed(SEED)
    # losses_NA, avg_losses_NA = sportisse_NA()
    # all_losses.append(losses_NA)
    # all_avg_losses.append(avg_losses_NA)
    # all_labels.append("SportisseNA")
    #
    # np.random.seed(SEED)
    # losses_sportisseRdk, avg_losses_sportisseRdk = sportisse_Rdk()
    # all_losses.append(losses_sportisseRdk)
    # all_avg_losses.append(avg_losses_sportisseRdk)
    # all_labels.append("SportisseRdk")

    np.random.seed(SEED)
    losses_Rdk, avg_losses_Rdk = compression_SGD(sparsificator)
    all_losses.append(losses_Rdk)
    all_avg_losses.append(avg_losses_Rdk)
    all_labels.append("Rdk")

    np.random.seed(SEED)
    losses_Qtz, avg_losses_Qtz = compression_SGD(quantizator)
    all_losses.append(losses_Qtz)
    all_avg_losses.append(avg_losses_Qtz)
    all_labels.append("Qtz")

    np.random.seed(SEED)
    losses_vanilla, avg_losses_vanilla = compression_SGD(SQuantization(0, DIM))
    all_losses.append(losses_vanilla)
    all_avg_losses.append(avg_losses_vanilla)
    all_labels.append("Vanilla SGD")

    optimal_loss = compute_empirical_risk(W_STAR)

    setup_plot_with_SGD(all_losses, all_avg_losses, all_labels, optimal_loss)



