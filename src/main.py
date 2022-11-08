"""
Created by Constantin Philippenko, 20th December 2021.
"""
import argparse

import numpy as np

import matplotlib

from src.PlotUtils import plot_only_avg, setup_plot_with_SGD, plot_eigen_values
from src.Utilities import create_folder_if_not_existing
from src.federated_learning.Client import Client, check_clients

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

from src.SGD import SGDRun, SeriesOfSGD, SGDVanilla, SGDCompressed

DIM = 100
POWER_COV = 4
R_SIGMA=0

DECR_STEP_SIZE = False
EIGENVALUES = None

BATCH_SIZE = 1

DO_LOGISTIC_REGRESSION = False

STOCHASTIC = True

NB_RUNS = 5

step_size = lambda it, r2, omega: 1 / (2 * (omega + 1) * r2)


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
        default=10**5,
    )
    parser.add_argument(
        "--use_ortho_matrix",
        type=bool,
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
    args = parser.parse_args()
    dataset_size = args.dataset_size
    nb_clients = int(args.nb_clients)
    use_ortho_matrix = True if args.use_ortho_matrix == "True" else False
    heterogeneity = args.heterogeneity
    

    np.random.seed(10)
    clients = [Client(i, DIM, dataset_size // nb_clients, POWER_COV, nb_clients, use_ortho_matrix, heterogeneity)
               for i in range(nb_clients)]

    sgd_series = SeriesOfSGD()
    for run_id in range(NB_RUNS):
        check_clients(clients, heterogeneity)
        synthetic_dataset = clients[0].dataset
        hash_string = synthetic_dataset.string_for_hash(STOCHASTIC)

        labels = ["no compr.", "1-quantiz.", "sparsif.", "sketching", "rand-1", "partial part."]

        w_star = np.mean([client.dataset.w_star for client in clients], axis=0)

        vanilla_sgd = SGDVanilla(clients, step_size, sto=STOCHASTIC, batch_size=BATCH_SIZE)
        sgd_nocompr = vanilla_sgd.gradient_descent(label=labels[0])
        all_sgd = [sgd_nocompr]

        my_compressors = [synthetic_dataset.quantizator, synthetic_dataset.sparsificator, synthetic_dataset.sketcher,
                          synthetic_dataset.rand1, synthetic_dataset.all_or_nothinger]

        for i in range(len(my_compressors)):
            compressor = my_compressors[i]
            print("Compressor: {0}".format(compressor.get_name()))
            all_sgd.append(
                SGDCompressed(clients, step_size, compressor, sto=STOCHASTIC, batch_size=BATCH_SIZE).gradient_descent(
                label=labels[i+1]))

        optimal_loss = 0
        print("Optimal loss:", optimal_loss)
        sgd_series.append(all_sgd)
        create_folder_if_not_existing("pickle")
        sgd_series.save("pickle/C{0}-{1}".format(nb_clients, hash_string))

        for client in clients:
            client.regenerate_dataset()

    plot_only_avg(sgd_series, optimal_loss=optimal_loss,
                  hash_string="C{0}-{1}".format(nb_clients, hash_string))

    setup_plot_with_SGD(sgd_series, optimal_loss=optimal_loss,
                        hash_string="C{0}-{1}_both".format(nb_clients, hash_string))

    plot_eigen_values(sgd_series, hash_string="C{0}-{1}".format(nb_clients, hash_string))
