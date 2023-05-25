"""
Created by Constantin Philippenko, 20th December 2021.
"""
import argparse

import matplotlib
import numpy as np

from src.federated_learning.Client import Client, check_clients
from src.utilities.PlotUtils import plot_only_avg, setup_plot_with_SGD, plot_eigen_values
from src.utilities.Utilities import create_folder_if_not_existing

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

from src.SGD import SeriesOfSGD, SGDVanilla, SGDCompressed

DIM = 100
R_SIGMA=0

DECR_STEP_SIZE = False
EIGENVALUES = None

BATCH_SIZE = 1

DO_LOGISTIC_REGRESSION = False

STOCHASTIC = True

NB_RUNS = 1


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
        default=5*10**4,
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
        "--gamma_horizon",
        type=str,
        help="If True, step size set to K^{-2/5}",
        required=False,
        default=True,
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
    gamma_horizon = True if args.gamma_horizon == "True" else False
    heterogeneity = args.heterogeneity

    if gamma_horizon:
        step_size = lambda it, r2, omega, K: K ** (-2 / 5)
    else:
        step_size = lambda it, r2, omega, K: 1 / (2 * (omega + 1) * r2)

    np.random.seed(10)
    clients = [Client(i, DIM, size_dataset // nb_clients, power_cov, nb_clients, use_ortho_matrix, heterogeneity)
               for i in range(nb_clients)]

    sgd_series = SeriesOfSGD()
    for run_id in range(NB_RUNS):
        check_clients(clients, heterogeneity)
        synthetic_dataset = clients[0].dataset
        gamma = "horizon" if gamma_horizon else None
        hash_string="C{0}-{1}".format(nb_clients, synthetic_dataset.string_for_hash(NB_RUNS, STOCHASTIC, step=gamma,
                                                                                    reg=args.reg))

        labels = ["no compr.", "1-quantiz.", "sparsif.", "sketching",
                  r"rand-$h$", "partial part."]

        w_star = np.mean([client.dataset.w_star for client in clients], axis=0)

        vanilla_sgd = SGDVanilla(clients, step_size, sto=STOCHASTIC, batch_size=BATCH_SIZE, reg=reg)
        sgd_nocompr = vanilla_sgd.gradient_descent(label=labels[0])
        all_sgd = [sgd_nocompr]

        my_compressors = [synthetic_dataset.quantizator, synthetic_dataset.sparsificator, synthetic_dataset.sketcher,
                          synthetic_dataset.rand1, synthetic_dataset.all_or_nothinger]

        for i in range(len(my_compressors)):
            compressor = my_compressors[i]
            print("Compressor: {0}".format(compressor.get_name()))
            all_sgd.append(
                SGDCompressed(clients, step_size, compressor, sto=STOCHASTIC, batch_size=BATCH_SIZE, reg=reg).gradient_descent(
                label=labels[i+1]))

        optimal_loss = 0
        print("Optimal loss:", optimal_loss)
        sgd_series.append(all_sgd)
        create_folder_if_not_existing("pickle")
        sgd_series.save("pickle/C{0}-{1}".format(nb_clients, hash_string))

        plot_only_avg(sgd_series, optimal_loss=optimal_loss, hash_string=hash_string)

        for client in clients:
            client.regenerate_dataset()

    setup_plot_with_SGD(sgd_series, optimal_loss=optimal_loss,
                        hash_string="{0}_both".format(hash_string))

    plot_eigen_values(sgd_series, hash_string=hash_string)
