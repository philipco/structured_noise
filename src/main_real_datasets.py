"""
Created by Constantin Philippenko, 20th December 2021.
"""
import argparse

import numpy as np

import matplotlib

from src.RealDataset import RealLifeDataset, split_across_clients
from src.utilities.PickleHandler import pickle_saver
from src.utilities.PlotUtils import plot_only_avg, setup_plot_with_SGD, plot_eigen_values
from src.utilities.Utilities import create_folder_if_not_existing
from src.federated_learning.Client import Client, check_clients, ClientRealDataset

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

from src.SGD import SeriesOfSGD, SGDVanilla, SGDCompressed, compute_wstar

nb_clients = 1

EPOCHS = 20

DECR_STEP_SIZE = False
EIGENVALUES = None

BATCH_SIZE = 16

DO_LOGISTIC_REGRESSION = False

STOCHASTIC = True

NB_RUNS = 2

NB_CLIENTS = 10

step_size = lambda it, r2, omega, K: 1 / (2 * (omega + 1) * r2)

if __name__ == '__main__':

    np.random.seed(10)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Dataset name",
        required=True,
    )
    parser.add_argument(
        "--nb_clients",
        type=int,
        default=10,
        help="Number of clients",
        required=False,
    )
    args = parser.parse_args()
    dataset_name = args.dataset_name
    nb_clients = args.nb_clients
    s = 16 if dataset_name == "cifar10" else 8

    real_datasets = [RealLifeDataset(dataset_name, s=s)]
    if real_datasets[0].w_star is None:
        compute_wstar(real_datasets[0], step_size, BATCH_SIZE)
        pickle_saver(real_datasets[0], "pickle/real_dataset/{0}".format(dataset_name))

    if nb_clients > 1:
        real_datasets = split_across_clients(real_datasets[0], nb_clients)
    clients = [ClientRealDataset(i, real_datasets[i].dim, real_datasets[i].size_dataset, real_datasets[i])
               for i in range(nb_clients)]

    labels = ["no compr.", r"$s$-quantiz.", "sparsif.", "sketching", r"rand-$h$", "partial part."]

    sgd_series = SeriesOfSGD()
    for run_id in range(NB_RUNS):
        hash_string = real_datasets[0].string_for_hash(NB_RUNS, STOCHASTIC)

        vanilla_sgd = SGDVanilla(clients, step_size, sto=STOCHASTIC, batch_size=BATCH_SIZE, nb_epoch=EPOCHS)
        sgd_nocompr = vanilla_sgd.gradient_descent(label=labels[0])
        all_sgd = [sgd_nocompr]

        my_compressors = [real_datasets[0].quantizator, real_datasets[0].sparsificator, real_datasets[0].sketcher,
                          real_datasets[0].rand1, real_datasets[0].all_or_nothinger]

        for i in range(len(my_compressors)):
            compressor = my_compressors[i]
            print("Compressor: {0}".format(compressor.get_name()))
            all_sgd.append(
                SGDCompressed(clients, step_size, compressor, sto=STOCHASTIC, batch_size=BATCH_SIZE, nb_epoch=EPOCHS).gradient_descent(
                    label=labels[i + 1]))

        optimal_loss = 0
        print("Optimal loss:", optimal_loss)
        sgd_series.append(all_sgd)
        create_folder_if_not_existing("pickle")
        sgd_series.save("pickle/C{0}-{1}".format(nb_clients, hash_string))

        plot_only_avg(sgd_series, optimal_loss=optimal_loss,
                      hash_string="C{0}-{1}".format(nb_clients, hash_string))

        for client in clients:
            client.regenerate_dataset()

    setup_plot_with_SGD(sgd_series, optimal_loss=optimal_loss,
                        hash_string="C{0}-{1}_both".format(nb_clients, hash_string))

    plot_eigen_values(sgd_series, hash_string="C{0}-{1}".format(nb_clients, hash_string))
