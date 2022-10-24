"""
Created by Constantin Philippenko, 20th December 2021.
"""
import copy
import hashlib
from typing import List

import numpy as np

import matplotlib

from src.PlotUtils import plot_only_avg, setup_plot_with_SGD, FONTSIZE, plot_eigen_values
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

SIZE_DATASET = 10**3
DIM = 100
POWER_COV = 4 # 1 for wstar
R_SIGMA=0
NB_CLIENTS = 1

DECR_STEP_SIZE = False
EIGENVALUES = None

NB_EPOCH = 400
BATCH_SIZE = 1 # 64 for wstar

USE_ORTHO_MATRIX = True
DO_LOGISTIC_REGRESSION = False

HETEROGENEITY = "homog" # "wstar" "sigma" "homog"
STOCHASTIC = True

NB_RUNS = 5

step_size = lambda it, r2, omega: 1 / (2 * (omega + 1) * r2)


if __name__ == '__main__':

    np.random.seed(10)
    clients = [Client(i, DIM, SIZE_DATASET // NB_CLIENTS, POWER_COV, NB_CLIENTS, USE_ORTHO_MATRIX, HETEROGENEITY)
               for i in range(NB_CLIENTS)]

    sgd_series = SeriesOfSGD()
    for run_id in range(NB_RUNS):
        check_clients(clients, HETEROGENEITY)
        synthetic_dataset = clients[0].dataset
        synthetic_dataset.power_cov = POWER_COV

        hash_string = hashlib.shake_256(clients[0].dataset.string_for_hash().encode()).hexdigest(4)

        labels = ["no compr.", "1-quantiz.", "sparsif.", "sketching", "rand-1", "partial part."]

        w_star = np.mean([client.dataset.w_star for client in clients], axis=0)

        vanilla_sgd = SGDVanilla(clients, step_size, nb_epoch=NB_EPOCH, sto=STOCHASTIC, batch_size=BATCH_SIZE)
        sgd_nocompr = vanilla_sgd.gradient_descent(label=labels[0], deacreasing_step_size=DECR_STEP_SIZE)
        all_sgd = [sgd_nocompr]

        my_compressors = [synthetic_dataset.quantizator, synthetic_dataset.sparsificator, synthetic_dataset.sketcher,
                          synthetic_dataset.rand1, synthetic_dataset.all_or_nothinger]

        for i in range(len(my_compressors)):
            compressor = my_compressors[i]
            print("Compressor: {0}".format(compressor.get_name()))
            all_sgd.append(
                SGDCompressed(clients, step_size, compressor, nb_epoch=NB_EPOCH, sto=STOCHASTIC, batch_size=BATCH_SIZE).gradient_descent(
                label=labels[i+1], deacreasing_step_size=DECR_STEP_SIZE))

        optimal_loss = 0
        print("Optimal loss:", optimal_loss)
        sgd_series.append(all_sgd)
        create_folder_if_not_existing("pickle")
        sgd_series.save("pickle/C{0}-{1}".format(NB_CLIENTS, synthetic_dataset.string_for_hash()))


        # test = SeriesOfSGD()
        # test.append(all_sgd)
        # plot_only_avg(test, optimal_loss=optimal_loss,
        #               hash_string="C{0}-{1}-T{2}".format(NB_CLIENTS, synthetic_dataset.string_for_hash(), run_id))

        for client in clients:
            client.regenerate_dataset()

    plot_only_avg(sgd_series, optimal_loss=optimal_loss,
                  hash_string="C{0}-{1}".format(NB_CLIENTS, synthetic_dataset.string_for_hash()))

    setup_plot_with_SGD(sgd_series, optimal_loss=optimal_loss,
                        hash_string="C{0}-{1}_both".format(NB_CLIENTS, synthetic_dataset.string_for_hash()))

    plot_eigen_values(sgd_series, hash_string="C{0}-{1}".format(NB_CLIENTS, synthetic_dataset.string_for_hash()))
