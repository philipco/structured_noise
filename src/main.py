"""
Created by Constantin Philippenko, 20th December 2021.
"""
import copy
import hashlib

import numpy as np

import matplotlib
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

from matplotlib import pyplot as plt

from src.CompressionModel import Sketching
from src.SGD import SGDRun, SeriesOfSGD, SGDVanilla, SGDCompressed
from src.SyntheticDataset import SyntheticDataset

SIZE_DATASET = 10**5
DIM = 100
POWER_COV = 2
R_SIGMA=0

EIGENVALUES = None #np.array([1,2])

USE_ORTHO_MATRIX = True
DO_LOGISTIC_REGRESSION = False

SEED = 25


def plot_SGD_and_AVG(axes, sgd_run: SGDRun, optimal_loss):

    axes[0].plot(np.log10(sgd_run.log_xaxis), np.log10(sgd_run.losses - optimal_loss),
                 label="SGD {0}".format(sgd_run.label))
    axes[1].plot(np.log10(sgd_run.log_xaxis), np.log10(sgd_run.avg_losses - optimal_loss),
                 label="AvgSGD {0}".format(sgd_run.label))


def setup_plot(losses1, avg_losses1, label1, losses2, avg_losses2, label2, optimal_loss):
    fig, ax = plt.subplots(figsize=(8, 7))
    plot_SGD_and_AVG(losses1, avg_losses1, optimal_loss, label1)
    plot_SGD_and_AVG(losses2, avg_losses2, optimal_loss, label2)

    ax.label(loc='best', fontsize=15)
    ax.set_xlabel(r"$\log_{10}(n)$", fontsize=15)
    ax[0].set_ylabel(r"$\log_{10}(F(w_k) - F(w_*))$", fontsize=15)
    ax[1].set_ylabel(r"$\log_{10}(F(\bar w_k) - F(w_*))$", fontsize=15)
    ax.grid(True)
    plt.show()


def setup_plot_with_SGD(all_sgd, sgd_nocompr: SGDRun, optimal_loss, hash_string: str = None):
    fig, axes = plt.subplots(2, figsize=(8, 7))

    plot_SGD_and_AVG(axes, sgd_nocompr, optimal_loss)

    for sgd_try in all_sgd:
        plot_SGD_and_AVG(axes, sgd_try, optimal_loss)

    for ax in axes:
        ax.legend(loc='best', fontsize=15)
        ax.set_ylim(top=0.5)
        ax.grid(True)
    axes[0].set_ylabel(r"$\log_{10}(F(w_k) - F(w_*))$", fontsize=15)
    axes[1].set_ylabel(r"$\log_{10}(F(\bar w_k) - F(w_*))$", fontsize=15)
    axes[1].set_xlabel(r"$\log_{10}(n)$", fontsize=15)

    if hash_string:
        plt.savefig('{0}.eps'.format("./pictures/" + hash_string), format='eps')
        plt.close()
    else:
        plt.show()

def plot_only_avg(all_sgd, sgd_nocompr: SGDRun, optimal_loss, hash_string: str = None):
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(np.log10(sgd_nocompr.log_xaxis), np.log10(sgd_nocompr.avg_losses - optimal_loss),
                 label="{0}".format(sgd_nocompr.label))

    for sgd_try in all_sgd:
        ax.plot(np.log10(sgd_try.log_xaxis), np.log10(sgd_try.avg_losses - optimal_loss),
                label="{0}".format(sgd_try.label))

    ax.legend(loc='best', fontsize=15)
    ax.set_ylim(top=0.5)
    ax.grid(True)
    ax.set_ylabel(r"$\log_{10}(F(\bar w_k) - F(w_*))$", fontsize=15)
    ax.set_xlabel(r"$\log_{10}(n)$", fontsize=15)
    ax.set_title("Avg SGD")

    if hash_string:
        plt.savefig('{0}.png'.format("./pictures/" + hash_string), bbox_inches='tight', dpi=600)
        plt.close()
    else:
        plt.show()


def plot_eigen_values(list_of_sgd, hash_string: str = None):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for sgd_try in list_of_sgd:
        plt.plot(np.log10(np.arange(1, DIM + 1)), np.log10(sgd_try.diag_cov_gradients), label=sgd_try.label, lw=2)
    ax.tick_params(axis='both', labelsize=15)
    ax.legend(loc='best', fontsize=15)
    ax.set_xlabel(r"$\log(i), \forall i \in \{1, ..., d\}$", fontsize=15)
    ax.set_ylabel(r"$\log(Diag(\frac{\mathcal C (X)^T.\mathcal C (X)}{n})_i)$", fontsize=15)
    plt.legend(loc='best', fontsize=15)
    if hash_string:
        plt.savefig('{0}-eigenvalues.eps'.format("./pictures/" + hash_string), format='eps')
        plt.close()
    else:
        plt.show()


if __name__ == '__main__':

    synthetic_dataset = SyntheticDataset()
    synthetic_dataset.generate_dataset(DIM, size_dataset=SIZE_DATASET, power_cov=POWER_COV, r_sigma=R_SIGMA,
                                       use_ortho_matrix=USE_ORTHO_MATRIX, do_logistic_regression=DO_LOGISTIC_REGRESSION,
                                       eigenvalues=EIGENVALUES)

    hash_string = hashlib.shake_256(synthetic_dataset.string_for_hash().encode()).hexdigest(4)

    # sgd_sportisse = SGDSportisse(copy.deepcopy(synthetic_dataset),
    #                              synthetic_dataset.sparsificator).gradient_descent(label="sportisse")

    # sparse_sketcher = Sketching(synthetic_dataset.LEVEL_RDK, synthetic_dataset.dim, randomized=True, type_proj="rdk")

    # sgd_rand_sketching_rdk = SGDCompressed(copy.deepcopy(synthetic_dataset), sparse_sketcher).gradient_descent(
    #     label="rand sketch rdk")

    vanilla_sgd = SGDVanilla(synthetic_dataset)
    sgd_nocompr = vanilla_sgd.gradient_descent(label="no compression", deacreasing_step_size=True)

    w_ERM = (np.linalg.pinv(synthetic_dataset.X_complete.T.dot(synthetic_dataset.X_complete))
             .dot(synthetic_dataset.X_complete.T)).dot(synthetic_dataset.Y)
    optimal_loss = vanilla_sgd.compute_true_risk(synthetic_dataset.w_star, synthetic_dataset.X_complete,
                                                      synthetic_dataset.Y)

    my_compressors = [synthetic_dataset.quantizator, synthetic_dataset.stabilized_quantizator, 
                      synthetic_dataset.rand_sketcher, synthetic_dataset.sparsificator, synthetic_dataset.rand1,
                      synthetic_dataset.all_or_nothinger]
    
    all_sgd = []
    for compressor in my_compressors:
        all_sgd.append(SGDCompressed(synthetic_dataset, compressor).gradient_descent(label=compressor.get_name(),
                                                                                     deacreasing_step_size=True))

    sgd_series = SeriesOfSGD(all_sgd)
    sgd_series.save("pickle/" + synthetic_dataset.string_for_hash())

    setup_plot_with_SGD(all_sgd, sgd_nocompr=sgd_nocompr, optimal_loss=optimal_loss,
                        hash_string=synthetic_dataset.string_for_hash())

    plot_only_avg(all_sgd, sgd_nocompr=sgd_nocompr, optimal_loss=optimal_loss,
                        hash_string=synthetic_dataset.string_for_hash())

    plot_eigen_values(all_sgd + [sgd_nocompr], hash_string=synthetic_dataset.string_for_hash())
