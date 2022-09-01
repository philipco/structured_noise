"""
Created by Constantin Philippenko, 20th December 2021.
"""
import copy
import hashlib

import numpy as np

import matplotlib

from src.PlotUtils import plot_SGD_and_AVG, plot_only_avg, setup_plot_with_SGD
from src.Utilities import create_folder_if_not_existing
from src.federated_learning.Client import Client, check_clients

from src.CompressionModel import NoCompression

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

from matplotlib import pyplot as plt

from src.SGD import SGDRun, SeriesOfSGD, SGDVanilla, SGDCompressed, SGDArtemis, FullGD

SIZE_DATASET = 100000
DIM = 100
POWER_COV = 4 # 1 for wstar
R_SIGMA=0
NB_CLIENTS = 10

DECR_STEP_SIZE = False
EIGENVALUES = None

NB_EPOCH = 400
BATCH_SIZE = 1 # 64 for wstar

USE_ORTHO_MATRIX = True
DO_LOGISTIC_REGRESSION = False

HETEROGENEITY = "homog" # "wstar" "sigma" "homog"
STOCHASTIC = True

step_size = lambda it, r2, omega: 1 / (2 * (omega + 1) * r2)

def setup_plot(losses1, avg_losses1, label1, losses2, avg_losses2, label2, optimal_loss):
    fig, ax = plt.subplots(figsize=(8, 7))
    plot_SGD_and_AVG(losses1, avg_losses1, optimal_loss, label1)
    plot_SGD_and_AVG(losses2, avg_losses2, optimal_loss, label2)

    ax.label(loc='best', fontsize=15)
    ax.set_xlabel(r"$\log_{10}(k)$", fontsize=15)
    ax[0].set_ylabel(r"$\log_{10}(F(w_k) - F(w_*))$", fontsize=15)
    ax[1].set_ylabel(r"$\log_{10}(F(\bar w_k) - F(w_*))$", fontsize=15)
    ax.grid(True)
    plt.show()


def plot_eigen_values(list_of_sgd, hash_string: str = None):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for sgd_try in list_of_sgd:
        plt.plot(np.log10(np.arange(1, DIM + 1)), np.log10(sgd_try.diag_cov_gradients), label=sgd_try.label, lw=2)
    ax.tick_params(axis='both', labelsize=15)
    ax.legend(loc='lower left', fontsize=15)
    ax.set_xlabel(r"$\log(i), \forall i \in \{1, ..., d\}$", fontsize=15)
    ax.set_ylabel(r"$\log(Diag(\frac{\mathcal C (X)^T.\mathcal C (X)}{n})_i)$", fontsize=15)
    plt.legend(loc='best', fontsize=15)
    if hash_string:
        plt.savefig('{0}-eigenvalues.pdf'.format("./pictures/" + hash_string), bbox_inches='tight', dpi=600)
        plt.close()
    else:
        plt.show()


if __name__ == '__main__':

    np.random.seed(10)
    clients = [Client(i, DIM, SIZE_DATASET // NB_CLIENTS, POWER_COV, NB_CLIENTS, USE_ORTHO_MATRIX, HETEROGENEITY)
               for i in range(NB_CLIENTS)]
    check_clients(clients, HETEROGENEITY)
    synthetic_dataset = clients[0].dataset
    synthetic_dataset.power_cov = POWER_COV

    hash_string = hashlib.shake_256(clients[0].dataset.string_for_hash().encode()).hexdigest(4)

    labels = ["no compr.", "1-quantiz.", "sparsif.", "sketching", "rand-1", "partial part."]

    w_star = np.mean([client.dataset.w_star for client in clients], axis=0)
    vanilla_sgd = SGDVanilla(clients, step_size, nb_epoch=NB_EPOCH, sto=STOCHASTIC, batch_size=BATCH_SIZE)
    sgd_nocompr = vanilla_sgd.gradient_descent(label=labels[0], deacreasing_step_size=DECR_STEP_SIZE)

    my_compressors = [synthetic_dataset.quantizator, synthetic_dataset.sparsificator, synthetic_dataset.sketcher,
                      synthetic_dataset.rand1, synthetic_dataset.all_or_nothinger]
    
    all_sgd = []
    for i in range(len(my_compressors)):
        compressor = my_compressors[i]
        print("Compressor: {0}".format(compressor.get_name()))
        all_sgd.append(
            SGDCompressed(clients, step_size, compressor, nb_epoch=NB_EPOCH, sto=STOCHASTIC, batch_size=BATCH_SIZE).gradient_descent(
            label=labels[i+1], deacreasing_step_size=DECR_STEP_SIZE))

    optimal_loss = 0
    print("Optimal loss:", optimal_loss)
    sgd_series = SeriesOfSGD(all_sgd)
    create_folder_if_not_existing("pickle")
    sgd_series.save("pickle/C{0}-{1}".format(NB_CLIENTS, synthetic_dataset.string_for_hash()))

    setup_plot_with_SGD(all_sgd, sgd_nocompr=sgd_nocompr, optimal_loss=optimal_loss,
                        hash_string="C{0}-{1}_both".format(NB_CLIENTS, synthetic_dataset.string_for_hash()))

    plot_only_avg(all_sgd, sgd_nocompr=sgd_nocompr, optimal_loss=optimal_loss,
                  hash_string="C{0}-{1}".format(NB_CLIENTS, synthetic_dataset.string_for_hash()))

    plot_eigen_values([sgd_nocompr] + all_sgd, hash_string="C{0}-{1}".format(NB_CLIENTS, synthetic_dataset.string_for_hash()))
