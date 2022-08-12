"""Created by Constantin Philippenko, 7th April 2022."""
import random

import matplotlib
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.PlotUtils import confidence_ellipse, create_gif
from src.CompressionModel import *
from src.SyntheticDataset import SyntheticDataset
from src.Utilities import create_folder_if_not_existing

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

SIZE_DATASET = 100
POWER_COV = 4
R_SIGMA=0

DIM = 2
NB_CLIENTS = 1

FONTSIZE = 15
LINESIZE = 3

USE_ORTHO_MATRIX = True
HETEROGENEITY = "homog"

NB_RANDOM_COMPRESSION = 5

EIGENVALUES = np.array([1,10])
FOLDER = "pictures/ellipse/muL={0}/".format(min(EIGENVALUES)/max(EIGENVALUES))
create_folder_if_not_existing(FOLDER)

COLORS = ["tab:blue", "tab:orange", "tab:brown", "tab:green", "tab:red", "tab:purple", "tab:cyan"]


def plot_compressed_points(compressor, X, i, folder, ax_max):
    compressed_point_i = np.array([compressor.compress(X[i]) for j in range(NB_RANDOM_COMPRESSION)])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(X[:, 0], X[:, 1], color=COLORS[0], label="No compression")
    ax.scatter(compressed_point_i[:, 0], compressed_point_i[:, 1], color=COLORS[1], s=80, label="Compression")
    plt.plot(X[i][0], X[i][1], marker="o", markersize=10, markerfacecolor="red", label="Point of interest")
    ax.set_xlim(-ax_max, ax_max)
    ax.set_ylim(-ax_max, ax_max)
    plt.axvline(x=0, color="black")
    plt.axhline(y=0, color="black")
    plt.legend(loc='upper right', fontsize=FONTSIZE)
    plt.savefig("{0}/{1}.png".format(folder, i), dpi=70) # default is 100, to reduce how many pixels the figures has.
    plt.close()

    return compressed_point_i


def plot_some_random_compressed_points(X, all_compressed_point, filename, ax_max):
    fig, axis = plt.subplots(3, 3, figsize=(10, 8))
    axis_flat = axis.flat
    sampled_indices = random.sample(range(len(all_compressed_point)), 9)
    for idx in range(len(axis_flat)):
        ax = axis_flat[idx]
        i = sampled_indices[idx]
        ax.scatter(X[:, 0], X[:, 1], color=COLORS[0], alpha=0.5, label="No compression")
        ax.scatter(all_compressed_point[i][:, 0], all_compressed_point[i][:, 1], color=COLORS[1], label="Compression")
        ax.plot(X[i][0], X[i][1], marker="o", markersize=6, markerfacecolor="red", label="Point of interest")
        ax.set_xlim(-ax_max, ax_max)
        ax.set_ylim(-ax_max, ax_max)
        ax.axvline(x=0, color="black", lw=1)
        ax.axhline(y=0, color="black", lw=1)
    axis_flat[0].legend(loc='best')
    if USE_ORTHO_MATRIX:
        filename += "-ortho"
    plt.savefig("{0}.pdf".format(filename), bbox_inches='tight', dpi=600)  # default is 100, to reduce how many pixels the figures has.
    plt.close()


def compute_quadratic_error(x, all_compression):
    return np.mean([(c - x)**2 for c in all_compression])


def add_scatter_plot_to_figure(ax, X, all_compressed_point, compressor, data_covariance, covariance, ax_max):
    ax.scatter(X[:, 0], X[:, 1], color=COLORS[0], alpha=0.5, label="No compression", zorder=3)
    ax.scatter(all_compressed_point[:, 0], all_compressed_point[:, 1], color=COLORS[1],  alpha=0.5, s=25,
               label="Compression", zorder=1)
    #(covariance, "", ax, edgecolor=COLORS[1], zorder=2, lw = LINESIZE)
    # confidence_ellipse(data_covariance, "", ax, edgecolor=COLORS[0], zorder=2, lw=LINESIZE)
    ax.set_xlim(-ax_max, ax_max)
    ax.set_ylim(-ax_max, ax_max)
    ax.axvline(x=0, color="black")
    ax.axhline(y=0, color="black")
    ax.set_title(compressor.get_name(), fontsize=FONTSIZE)


def plot_compression_process_by_compressor(dataset, compressor, data_covariance, covariance, ax, ax_mse0, ax_mse1):

    X = dataset.X_complete
    ax_max = max([X[i,j] for i in range(X.shape[0]) for j in range(2)]) / dataset.LEVEL_RDK + 1 # (to see the entire ellipse).

    folder = FOLDER + compressor.get_name()
    create_folder_if_not_existing(folder)

    all_compressed_point = []
    for i in tqdm(range(SIZE_DATASET)):
        all_compressed_point.append(plot_compressed_points(compressor, X, i, folder, ax_max))

    plot_some_random_compressed_points(X, all_compressed_point, folder, ax_max)
    all_compressed_point = np.concatenate(all_compressed_point)
    add_scatter_plot_to_figure(ax, X, all_compressed_point, compressor, data_covariance, covariance, ax_max)

    filenames = ["{0}/{1}.png".format(folder, i) for i in range(SIZE_DATASET)]
    gif_name = "{0}-ortho".format(folder) if USE_ORTHO_MATRIX else folder
    create_gif(file_names=filenames, gif_name=gif_name + ".gif")


def compute_covariance(dataset, compressor, non_gaussian = True):
    X = dataset.X_complete

    X_compressed = X.copy()

    for i in range(SIZE_DATASET):
        # TODO : with gaussian multiplication to check that the distribution is still ...
        X_compressed[i] = compressor.compress(X[i])

    cov_matrix = X_compressed.T.dot(X_compressed) / SIZE_DATASET

    return cov_matrix, X_compressed


def get_all_covariances(dataset: SyntheticDataset, my_compressors):

    all_covariances = []
    all_compressed_points = []
    for compressor in tqdm(my_compressors):
        cov_matrix, compressed_points = compute_covariance(dataset, compressor)
        all_covariances.append(cov_matrix)
        all_compressed_points.append(compressed_points)

    return all_covariances, all_compressed_points


def plot_compression_process(dataset, my_compressors, covariances, labels):

    fig_distrib, axes_distrib = plt.subplots(2, 3, figsize=(10, 6))
    axes_distrib = [axes_distrib[0,0], axes_distrib[0,1], axes_distrib[0,2], axes_distrib[1,0], axes_distrib[1,1], axes_distrib[1,2]]

    fig_MSE, axes_MSE = plt.subplots(1, 2, figsize=(12, 9))

    for i in range(1, len(my_compressors)):
        compressor = my_compressors[i]
        plot_compression_process_by_compressor(dataset, compressor, covariances[0], covariances[i], axes_distrib[i - 1], axes_MSE[0],
                                                axes_MSE[1])
    axes_distrib[0].legend(loc='lower right', fontsize=FONTSIZE)

    filename = FOLDER + "scatter_plot"
    if USE_ORTHO_MATRIX:
        filename = "{0}-ortho".format(filename)
    fig_distrib.savefig("{0}.pdf".format(filename), bbox_inches='tight', dpi=600)


def plot_ellipse(dataset, covariances, labels):
    fig, ax = plt.subplots(figsize=(6, 6))
    for i in range(len(covariances)):
        ax.axvline(c='grey', lw=1)
        ax.axhline(c='grey', lw=1)
        confidence_ellipse(covariances[i], labels[i], ax, edgecolor=COLORS[i], zorder=0)

    ax.scatter(dataset.X_complete[:,0], dataset.X_complete[:,1], s=0.5)
    ax.scatter(0, 0, c='red', s=3)
    ax.set_title("Ellipse")
    ax.legend(fancybox=True, framealpha=0.5, fontsize=FONTSIZE)
    ax.axis('equal')

    # fig.subplots_adjust(hspace=0.25)
    filename = FOLDER + "ellipse"
    if USE_ORTHO_MATRIX:
        filename = "{0}-ortho".format(filename)
    plt.savefig("{0}.pdf".format(filename), bbox_inches='tight',dpi=600)


if __name__ == '__main__':

    synthetic_dataset = SyntheticDataset()
    synthetic_dataset.generate_constants(DIM, SIZE_DATASET, POWER_COV, R_SIGMA, NB_CLIENTS, USE_ORTHO_MATRIX,
                                         eigenvalues=EIGENVALUES, heterogeneity=HETEROGENEITY, w0_seed=None)
    synthetic_dataset.define_compressors()
    synthetic_dataset.generate_X()

    labels = ["no compr.", "quantiz.", "stab. quantiz.", "gauss. proj.", "sparsif", "rand1", "all or noth."]
    no_compressor = SQuantization(0, dim=DIM)
    my_compressors = [no_compressor, synthetic_dataset.quantizator, synthetic_dataset.stabilized_quantizator,
                      synthetic_dataset.rand_sketcher,
                      synthetic_dataset.sparsificator, synthetic_dataset.rand1,
                      synthetic_dataset.all_or_nothinger]

    all_covariances, all_compressed_points = get_all_covariances(synthetic_dataset, my_compressors)
    print("Printing all covariance matrices.")
    for i in range(len(labels)):
        print(labels[i])
        print(all_covariances[i])

    plot_compression_process(dataset=synthetic_dataset, my_compressors=my_compressors, covariances=all_covariances,
                             labels=labels)

    plot_ellipse(dataset=synthetic_dataset, covariances=all_covariances, labels=labels)