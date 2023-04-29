"""
Created by Constantin Philippenko, 20th December 2021.
"""
import argparse

import numpy as np

import matplotlib

from src.RealDataset import RealLifeDataset, split_across_clients
from src.utilities.PickleHandler import pickle_saver
from src.utilities.PlotUtils import plot_only_avg, setup_plot_with_SGD, plot_eigen_values
from src.utilities.Utilities import create_folder_if_not_existing, file_exist
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

BATCH_SIZE = 32

DO_LOGISTIC_REGRESSION = False

STOCHASTIC = True

NB_RUNS = 5

NB_CLIENTS = 10

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
    parser.add_argument(
        "--gamma_horizon",
        type=str,
        help="Only when two clients or more. Possible values: 'wstar', 'sigma', 'homog'",
        required=False,
        default=True,
    )
    parser.add_argument(
        "--reg",
        type=int,
        help="Only when two clients or more. Possible values: 'wstar', 'sigma', 'homog'",
        required=False,
        default=0,
    )
    parser.add_argument(
        "--heterogeneity",
        type=str,
        help="Only when two clients or more. Possible values: 'wstar', 'sigma', 'homog'",
        required=False,
        default=None,
    )
    args = parser.parse_args()
    dataset_name = args.dataset_name
    nb_clients = args.nb_clients
    gamma_horizon = True if args.gamma_horizon == "True" else False
    reg = args.reg if args.reg == 0 else 10**-args.reg
    heterogeneity = args.heterogeneity
    s = 16 if dataset_name == "cifar10" else 8

    if gamma_horizon:
        step_size = lambda it, r2, omega, K: K ** (-2 / 5)
    else:
        step_size = lambda it, r2, omega, K: 1 / (2 * (omega + 1) * r2)

    real_datasets = [RealLifeDataset(dataset_name, s=s)]

    if nb_clients > 1:
        real_datasets = split_across_clients(real_datasets[0], nb_clients, heterogeneity)
    clients = [ClientRealDataset(i, real_datasets[i].dim, real_datasets[i].size_dataset, real_datasets[i])
               for i in range(nb_clients)]
    if not file_exist("pickle/real_dataset/C{0}-{1}_wstar.pkl".format(nb_clients, dataset_name)):
        w_star = compute_wstar(clients, lambda it, r2, omega, K: 1 / (2 * (omega + 1) * r2), BATCH_SIZE)
        for c in clients:
            c.dataset.w_star = w_star
        pickle_saver(w_star, "pickle/real_dataset/C{0}-{1}_wstar".format(nb_clients, dataset_name))

    labels = ["no compr.", r"$s$-quantiz.", "sparsif.", "sketching", r"rand-$h$", "partial part."]

    sgd_series = SeriesOfSGD()
    for run_id in range(NB_RUNS):
        gamma = "horizon" if gamma_horizon else None
        hash_string = real_datasets[0].string_for_hash(NB_RUNS, STOCHASTIC, step=gamma, reg=args.reg, heterogeneity=heterogeneity)

        vanilla_sgd = SGDVanilla(clients, step_size, sto=STOCHASTIC, batch_size=BATCH_SIZE, nb_epoch=EPOCHS, reg=reg)
        sgd_nocompr = vanilla_sgd.gradient_descent(label=labels[0])
        all_sgd = [sgd_nocompr]

        my_compressors = [real_datasets[0].quantizator, real_datasets[0].sparsificator, real_datasets[0].sparsificator,
                          real_datasets[0].rand1, real_datasets[0].all_or_nothinger]

        for i in range(len(my_compressors)):
            compressor = my_compressors[i]
            print("Compressor: {0}".format(compressor.get_name()))
            all_sgd.append(
                SGDCompressed(clients, step_size, compressor, sto=STOCHASTIC, batch_size=BATCH_SIZE, nb_epoch=EPOCHS,
                              reg=reg).gradient_descent(label=labels[i + 1]))

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
