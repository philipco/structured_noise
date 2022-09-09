"""
Created by Constantin Philippenko, 1st September 2022.
"""
import hashlib

import numpy as np
from matplotlib.lines import Line2D

from src.PlotUtils import plot_only_avg, setup_plot_with_SGD
from src.SGD import SeriesOfSGD, SGDArtemis, SGDCompressed, SGDVanilla
from src.Utilities import create_folder_if_not_existing
from src.federated_learning.Client import check_clients, Client
from src.main import plot_eigen_values

SIZE_DATASET = 10**8
DIM = 100
POWER_COV = 4 # 1 for wstar
R_SIGMA=0
NB_CLIENTS = 10

DECR_STEP_SIZE = False
EIGENVALUES = None

NB_EPOCH = 400
BATCH_SIZE = 64 # 64 for wstar

USE_ORTHO_MATRIX = True
DO_LOGISTIC_REGRESSION = False

HETEROGENEITY = "wstar"
STOCHASTIC = True

step_size = lambda it, r2, omega: 1 / (2*r2)

if __name__ == '__main__':

    np.random.seed(10)
    clients = [Client(i, DIM, SIZE_DATASET // NB_CLIENTS, POWER_COV, NB_CLIENTS, USE_ORTHO_MATRIX, HETEROGENEITY)
               for i in range(NB_CLIENTS)]
    check_clients(clients, HETEROGENEITY)
    synthetic_dataset = clients[0].dataset
    synthetic_dataset.power_cov = POWER_COV

    hash_string = synthetic_dataset.string_for_hash(STOCHASTIC)

    labels = ["no compr.", "1-quantiz.", "sparsif.", "sketching", "rand-1", "partial part."]

    w_star = np.mean([client.dataset.w_star for client in clients], axis=0)
    vanilla_sgd = SGDVanilla(clients, step_size, nb_epoch=NB_EPOCH, sto=STOCHASTIC, batch_size=BATCH_SIZE)
    sgd_nocompr = vanilla_sgd.gradient_descent(label=labels[0], deacreasing_step_size=DECR_STEP_SIZE)

    my_compressors = [synthetic_dataset.quantizator, synthetic_dataset.sparsificator,
                      # synthetic_dataset.all_or_nothinger]
                      synthetic_dataset.sketcher, synthetic_dataset.rand1, synthetic_dataset.all_or_nothinger]

    all_sgd = []
    for i in range(len(my_compressors)):
        compressor = my_compressors[i]
        print("Compressor: {0}".format(compressor.get_name()))
        all_sgd.append(
            SGDCompressed(clients, step_size, compressor, nb_epoch=NB_EPOCH, sto=STOCHASTIC,
                          batch_size=BATCH_SIZE).gradient_descent(
                label=labels[i + 1], deacreasing_step_size=DECR_STEP_SIZE))
        all_sgd.append(
            SGDArtemis(clients, step_size, compressor, nb_epoch=NB_EPOCH, sto=STOCHASTIC, batch_size=BATCH_SIZE).gradient_descent(
                label=compressor.get_name() + "-art", deacreasing_step_size=DECR_STEP_SIZE))

    optimal_loss = 0
    print("Optimal loss:", optimal_loss)
    sgd_series = SeriesOfSGD(all_sgd)
    create_folder_if_not_existing("pickle")
    sgd_series.save("pickle/C{0}-{1}".format(NB_CLIENTS, hash_string))

    legend_line = [Line2D([0], [0], color="black", lw=2, label='w.o. mem.'),
                    Line2D([0], [0], linestyle="--", color="black", lw=2, label='w. mem.')]

    setup_plot_with_SGD(all_sgd, sgd_nocompr=sgd_nocompr, optimal_loss=optimal_loss,
                        hash_string="C{0}-{1}-artemis-_both".format(NB_CLIENTS, hash_string))
                        #custom_legend=legend_line)

    plot_only_avg(all_sgd, sgd_nocompr=sgd_nocompr, optimal_loss=optimal_loss,
                  hash_string="C{0}-{1}-artemis".format(NB_CLIENTS, hash_string),
                  custom_legend=legend_line)

    plot_eigen_values([sgd_nocompr] + all_sgd,
                      hash_string="C{0}-{1}-artemis".format(NB_CLIENTS, hash_string),
                      custom_legend=legend_line)
