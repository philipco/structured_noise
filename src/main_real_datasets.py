"""
Created by Constantin Philippenko, 20th December 2021.
"""
import argparse

import numpy as np

import matplotlib
from matplotlib.lines import Line2D

from src.RealDataset import RealLifeDataset, split_across_clients
from src.utilities.PickleHandler import pickle_saver, pickle_loader
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

from src.SGD import SeriesOfSGD, SGDVanilla, SGDCompressed, compute_wstar, SGDArtemis

EPOCHS = 50

DECR_STEP_SIZE = False
EIGENVALUES = None

BATCH_SIZE = 32
STOCHASTIC = True
NB_RUNS = 1

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
        help="If True, step size set to K^{-2/5}",
        required=False,
        default=True,
    )
    parser.add_argument(
        "--reg",
        type=int,
        help="Regularization. Parameters a leads to a reg coefficient of 10^-a",
        required=False,
        default=0,
    )
    parser.add_argument(
        "--heterogeneity",
        type=str,
        help="Only when two clients or more. Possible values: 'random', 'dirichlet', 'tsne'",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--use_memory",
        type=str,
        help="If True, use memory",
        required=False,
        default=True,
    )
    args = parser.parse_args()
    dataset_name = args.dataset_name
    nb_clients = args.nb_clients
    EPOCHS *= nb_clients
    gamma_horizon = True if args.gamma_horizon == "True" else False
    reg = args.reg if args.reg == 0 else 10**-args.reg
    heterogeneity = args.heterogeneity
    use_memory = True if args.use_memory == "True" else False
    s = 16 if dataset_name == "cifar10" else 8

    if use_memory:
        legend_line = [Line2D([0], [0], color="black", lw=2, label='w.o. mem.'),
                       Line2D([0], [0], linestyle="--", color="black", lw=2, label='w. mem.')]
    else:
        legend_line = None

    if gamma_horizon:
        step_size = lambda it, r2, omega, K: K ** (-2 / 5)
    else:
        step_size = lambda it, r2, omega, K: 1 / (2 * (omega + 1) * r2)

    real_datasets = [RealLifeDataset(dataset_name, s=s)]

    if nb_clients > 1:
        real_datasets = split_across_clients(real_datasets[0], nb_clients, heterogeneity, dataset_name)
    clients = [ClientRealDataset(i, real_datasets[i].dim, real_datasets[i].size_dataset, real_datasets[i])
               for i in range(nb_clients)]
    if not file_exist("pickle/real_dataset/C{0}-{1}-{2}_wstar.pkl".format(nb_clients, dataset_name, heterogeneity)):
        w_star = compute_wstar(clients, lambda it, r2, omega, K: 1 / (2 * (omega + 1) * r2), BATCH_SIZE)
    else:
        w_star = pickle_loader("pickle/real_dataset/C{0}-{1}-{2}_wstar".format(nb_clients, dataset_name, heterogeneity))
    for c in clients:
        c.dataset.w_star = w_star
    pickle_saver(w_star, "pickle/real_dataset/C{0}-{1}-{2}_wstar".format(nb_clients, dataset_name, heterogeneity))

    labels = ["no compr.", r"$s$-quantiz.", "sparsif.", "sketching", r"rand-$h$", "partial part."]

    sgd_series = SeriesOfSGD()
    for run_id in range(NB_RUNS):
        gamma = "horizon" if gamma_horizon else None
        hash_string = real_datasets[0].string_for_hash(NB_RUNS, STOCHASTIC, step=gamma, reg=args.reg,
                                                       heterogeneity=heterogeneity, memory=use_memory)

        vanilla_sgd = SGDVanilla(clients, step_size, sto=STOCHASTIC, batch_size=BATCH_SIZE, nb_epoch=EPOCHS, reg=reg)
        sgd_nocompr = vanilla_sgd.gradient_descent(label=labels[0])
        all_sgd = [sgd_nocompr]

        my_compressors = [real_datasets[0].quantizator, real_datasets[0].sparsificator, real_datasets[0].sketcher,
                          real_datasets[0].rand1, real_datasets[0].all_or_nothinger]

        for i in range(len(my_compressors)):
            compressor = my_compressors[i]
            print("Compressor: {0}".format(compressor.get_name()))
            all_sgd.append(
                SGDCompressed(clients, step_size, compressor, sto=STOCHASTIC, batch_size=BATCH_SIZE, nb_epoch=EPOCHS,
                              reg=reg).gradient_descent(label=labels[i + 1]))
            if use_memory:
                all_sgd.append(
                    SGDArtemis(clients, step_size, compressor, sto=STOCHASTIC, batch_size=BATCH_SIZE, nb_epoch=EPOCHS,
                              reg=reg, start_averaging=0).gradient_descent(label=labels[i + 1] + "-art"))

        optimal_loss = 0
        print("Optimal loss:", optimal_loss)
        sgd_series.append(all_sgd)
        create_folder_if_not_existing("pickle")
        sgd_series.save("pickle/C{0}-{1}".format(nb_clients, hash_string))

        plot_only_avg(sgd_series, optimal_loss=optimal_loss,
                      custom_legend=legend_line, with_artemis=True, hash_string="C{0}-{1}".format(nb_clients, hash_string))

        for client in clients:
            client.regenerate_dataset()

    setup_plot_with_SGD(sgd_series, optimal_loss=optimal_loss,
                        hash_string="C{0}-{1}_both".format(nb_clients, hash_string),
                        custom_legend=legend_line, with_artemis=True)

    plot_eigen_values(sgd_series, hash_string="C{0}-{1}".format(nb_clients, hash_string),
                      custom_legend=legend_line)
