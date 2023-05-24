"""
Created by Constantin Philippenko, 1st May 2023.

Used to generate the figure in the paper which uses a horizon-dependent step_size. It runs SGD for K iterations, then
restarts with K*10 iterations and so on.
"""
import argparse
from typing import Dict

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from src.federated_learning.Client import Client, check_clients
from src.utilities.PickleHandler import pickle_saver
from src.utilities.Utilities import create_folder_if_not_existing

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

folder = "pictures/sgd_horizon/"
create_folder_if_not_existing(folder)

from src.SGD import SeriesOfSGD, SGDVanilla, SGDCompressed

FONTSIZE = 15
LINESIZE = 3

COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]

DIM = 100
R_SIGMA = 0
BATCH_SIZE = 1
STOCHASTIC = True

NB_RUNS = 5


def plot_sgd(dict_sgd: Dict[int], x_log_scale: np.ndarray, hash_string: str):
    fig, ax = plt.subplots(figsize=(8, 4))

    i = 0
    for label, list_of_sgd in dict_sgd.items():
        avg_losses = np.mean([np.log10(sgd) for sgd in list_of_sgd], axis=0)
        avg_losses_var = np.std([np.log10(sgd) for sgd in list_of_sgd], axis=0)
        label_avg_sgd = label
        line_style = "-"
        color = COLORS[i]
        ax.plot(x_log_scale, avg_losses, label=label_avg_sgd, lw=LINESIZE, linestyle=line_style, color=color,
                marker="h")
        plt.fill_between(x_log_scale, avg_losses - avg_losses_var, avg_losses + avg_losses_var, alpha=0.2, color=color)
        i += 1

    l1 = ax.legend(loc='best', fontsize=FONTSIZE)
    ax.add_artist(l1)

    ax.grid(True)
    ax.set_ylabel(r"$\log_{10}(F(\overline{w}_k) - F(w_*))$", fontsize=FONTSIZE)
    ax.set_xlabel(r"$\log_{10}(k)$", fontsize=FONTSIZE)

    if hash_string:
        plt.savefig('{0}.pdf'.format("{0}/{1}".format(folder, hash_string)), bbox_inches='tight', dpi=600)
        plt.close()
    else:
        plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nb_clients",
        type=int,
        help="Number of clients",
        required=True,
    )
    parser.add_argument(
        "--dataset_size",
        type=int,
        help="Size of the dataset.",
        required=False,
        default=10**6,
    )
    parser.add_argument(
        "--power_cov",
        type=int,
        help="Eigenvalues' decay of the covariance.",
        required=False,
        default=4,
    )
    parser.add_argument(
        "--use_ortho_matrix",
        type=str,
        help="Use an orthogonal matrix.",
        required=False,
        default=True,
    )
    parser.add_argument(
        "--heterogeneity",
        type=str,
        help="Only when two clients or more. Possible values: 'wstar', 'sigma', 'homog'",
        required=False,
        default="homog",
    )
    parser.add_argument(
        "--reg",
        type=int,
        help="Regularization - parameters 'a' leads to a reg coefficient of 10^-a",
        required=False,
        default=0,
    )
    args = parser.parse_args()
    size_dataset = args.dataset_size
    power_cov = args.power_cov
    nb_clients = args.nb_clients
    reg = args.reg if args.reg == 0 else 10**-args.reg
    use_ortho_matrix = True if args.use_ortho_matrix == "True" else False
    heterogeneity = args.heterogeneity

    step_size = lambda it, r2, omega, K: K ** (-2 / 5) # Optimal step-size according to our theorems.

    np.random.seed(10)

    x_log_scale = [i for i in range(1, int(np.log10(size_dataset)) + 1)]

    labels = ["no compr.", "1-quantiz.", "sparsif.", "sketching", r"rand-$h$", "partial part."]
    dict_sgd = {}
    sgd_series = SeriesOfSGD()
    for l in labels: dict_sgd[l] = []
    for run_id in range(NB_RUNS):

        for l in labels: dict_sgd[l].append([])
        for x in x_log_scale:
            clients = [Client(i, DIM, 10**x // nb_clients, power_cov, nb_clients, use_ortho_matrix, heterogeneity)
                       for i in range(nb_clients)]

            sgd_series = SeriesOfSGD()
            check_clients(clients, heterogeneity)
            synthetic_dataset = clients[0].dataset
            hash_string = "C{0}-{1}".format(nb_clients, synthetic_dataset.string_for_hash(NB_RUNS,
                                                                                          STOCHASTIC, step="horizon",
                                                                                          reg=args.reg))

            w_star = np.mean([client.dataset.w_star for client in clients], axis=0)

            vanilla_sgd = SGDVanilla(clients, step_size, sto=STOCHASTIC, batch_size=BATCH_SIZE, reg=reg)
            sgd_nocompr = vanilla_sgd.gradient_descent(label=labels[0])
            dict_sgd[labels[0]][-1].append(sgd_nocompr.avg_losses[-1])

            my_compressors = [synthetic_dataset.quantizator, synthetic_dataset.sparsificator,
                              synthetic_dataset.sketcher,
                              synthetic_dataset.rand1, synthetic_dataset.all_or_nothinger]

            for i in range(len(my_compressors)):
                compressor = my_compressors[i]
                print("Compressor: {0}".format(compressor.get_name()))
                sgd = SGDCompressed(clients, step_size, compressor, sto=STOCHASTIC, batch_size=BATCH_SIZE,
                                    reg=reg).gradient_descent(label=labels[i + 1])

                dict_sgd[labels[i + 1]][-1].append(sgd.avg_losses[-1])

            create_folder_if_not_existing("pickle")
            pickle_saver(dict_sgd, "pickle/C{0}-{1}-dict_sgd.pkl".format(nb_clients, hash_string))

            for client in clients:
                client.regenerate_dataset()

        plot_sgd(dict_sgd, x_log_scale, hash_string)
