"""
Created by Constantin Philippenko, 1st September 2022.
"""
import argparse

import numpy as np
from matplotlib.lines import Line2D

from src.PlotUtils import plot_only_avg, setup_plot_with_SGD
from src.SGD import SeriesOfSGD, SGDArtemis, SGDCompressed, SGDVanilla
from src.Utilities import create_folder_if_not_existing
from src.federated_learning.Client import check_clients, Client
from src.main import plot_eigen_values

DIM = 100
POWER_COV = 4
R_SIGMA=0

EIGENVALUES = None
USE_ORTHO_MATRIX = True
DO_LOGISTIC_REGRESSION = False

HETEROGENEITY = "wstar"

NB_RUNS = 5

step_size = lambda it, r2, omega: 1 / (2*r2)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nb_clients",
        type=int,
        help="Number of clients",
        required=True,
    )
    parser.add_argument(
        "--stochastic",
        type=str,
        help="Stochastic or full batch",
        required=True,
        default=True,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size when stochastic.",
        required=False,
        default=32,
    )
    parser.add_argument(
        "--use_ortho_matrix",
        type=str,
        help="Use an orthogonal matrix.",
        required=False,
        default=True,
    )
    parser.add_argument(
        "--noiseless",
        type=str,
        help="Use noise or not",
        required=False,
        default=False,
    )
    args = parser.parse_args()
    nb_clients = args.nb_clients
    batch_size = args.batch_size
    stochastic = True if args.stochastic == "True" else False
    noiseless = True if args.noiseless == "True" else False
    use_ortho_matrix = True if args.use_ortho_matrix == "True" else False

    legend_line = [Line2D([0], [0], color="black", lw=2, label='w.o. mem.'),
                    Line2D([0], [0], linestyle="--", color="black", lw=2, label='w. mem.')]

    # if stochastic:
    size_dataset = 5*10**7
    nb_epoch = 1

    lower_sigma = 0 if noiseless else None

    np.random.seed(10)
    clients = [Client(i, DIM, size_dataset // nb_clients, POWER_COV, nb_clients, USE_ORTHO_MATRIX, HETEROGENEITY,
                      lower_sigma=lower_sigma) for i in range(nb_clients)]

    sgd_series = SeriesOfSGD()
    for run_id in range(NB_RUNS):
        check_clients(clients, HETEROGENEITY)
        synthetic_dataset = clients[0].dataset
        synthetic_dataset.power_cov = POWER_COV

        hash_string = synthetic_dataset.string_for_hash(nb_runs=NB_RUNS,stochastic=stochastic, batch_size=batch_size,
                                                        noiseless=noiseless)

        labels = ["no compr.", "1-quantiz.", "sparsif.", "sketching", "rand-1", "partial part."]

        w_star = np.mean([client.dataset.w_star for client in clients], axis=0)
        vanilla_sgd = SGDVanilla(clients, step_size, nb_epoch=nb_epoch, sto=stochastic, batch_size=batch_size,
                                 start_averaging=0)
        sgd_nocompr = vanilla_sgd.gradient_descent(label=labels[0])
        all_sgd = [sgd_nocompr]

        my_compressors = [synthetic_dataset.quantizator, synthetic_dataset.sparsificator,
                          synthetic_dataset.sketcher, synthetic_dataset.rand1, synthetic_dataset.all_or_nothinger]

        for i in range(len(my_compressors)):
            compressor = my_compressors[i]
            print("Compressor: {0}".format(compressor.get_name()))
            all_sgd.append(
                SGDCompressed(clients, step_size, compressor, nb_epoch=nb_epoch, sto=stochastic,
                              batch_size=batch_size, start_averaging=0).gradient_descent(label=labels[i + 1]))
            all_sgd.append(
                SGDArtemis(clients, step_size, compressor, nb_epoch=nb_epoch, sto=stochastic,
                           batch_size=batch_size, start_averaging=0).gradient_descent(label=labels[i + 1] + "-art"))

        optimal_loss = 0
        print("Optimal loss:", optimal_loss)
        sgd_series.append(all_sgd)
        create_folder_if_not_existing("pickle")
        sgd_series.save("pickle/C{0}-{1}".format(nb_clients, hash_string))

        plot_only_avg(sgd_series, optimal_loss=optimal_loss,
                      hash_string="C{0}-{1}-artemis".format(nb_clients, hash_string),
                      custom_legend=legend_line, with_artemis=True, stochastic=stochastic)

        for client in clients:
            client.regenerate_dataset(lower_sigma)

    setup_plot_with_SGD(sgd_series, optimal_loss=optimal_loss,
                        hash_string="C{0}-{1}-artemis_both".format(nb_clients, hash_string),
                        custom_legend=legend_line, with_artemis=True)

    plot_eigen_values(sgd_series, hash_string="C{0}-{1}-artemis".format(nb_clients, hash_string),
                      custom_legend=legend_line)
