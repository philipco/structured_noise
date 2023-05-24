"""
Created by Constantin Philippenko, 7th April 2022.

Used to generate the figure in the paper which gives: (1) the scatter plot of the features w./w.o. compression (2) their
corresponding ellipses.
"""
import random
from typing import List

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from tqdm import tqdm

from src.CompressionModel import *
from src.SyntheticDataset import SyntheticDataset
from src.TheoreticalCov import get_theoretical_cov, compress_and_compute_covariance
from src.utilities.PlotUtils import confidence_ellipse, create_gif, COLORS, FONTSIZE, add_scatter_plot_to_figure
from src.utilities.Utilities import create_folder_if_not_existing

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

SIZE_DATASET = 10000
POWER_COV = 4
R_SIGMA=0

DIM = 2
NB_CLIENTS = 1

USE_ORTHO_MATRIX = True
HETEROGENEITY = "homog"

WITH_STANDARDISATION = True

NB_RANDOM_COMPRESSION = 1
NB_COMPRESSED_POINT = SIZE_DATASET

FEATURES_DISTRIBUTION = "normal" # cauchy, diamond or normal
EIGENVALUES = np.array([1,10])
if FEATURES_DISTRIBUTION in ["cauchy", "normal"]:
    FOLDER = "../pictures/ellipse/{0}/muL={1}/".format(FEATURES_DISTRIBUTION, min(EIGENVALUES)/max(EIGENVALUES))
else:
    FOLDER = "../pictures/ellipse/diamond/"
create_folder_if_not_existing(FOLDER)


def get_all_covariances(dataset: SyntheticDataset, my_compressors: List[CompressionModel]):
    """Return all compressors' covariances and compressed points."""

    all_covariances = []
    all_compressed_points = []
    for compressor in tqdm(my_compressors):
        cov_matrix, compressed_points = compress_and_compute_covariance(dataset, compressor)
        all_covariances.append(cov_matrix)
        all_compressed_points.append(compressed_points)

    return all_covariances, all_compressed_points


def plot_compressed_points(compressor: CompressionModel, X: np.ndarray, i: int, folder: str, ax_max: plt.Axes):
    """Plot the compressed points on the given axis."""
    compressed_point_i = np.array([compressor.compress(X[i]) for j in range(NB_RANDOM_COMPRESSION)])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(X[:250, 0], X[:250, 1], color=COLORS[0],  alpha=0.5, label="No compression")
    ax.scatter(compressed_point_i[:250, 0], compressed_point_i[:250, 1], color=COLORS[1], s=80, label="Compression")
    plt.plot(X[i][0], X[i][1], marker="o", markersize=10, markerfacecolor="red", label="Point of interest")
    ax.set_xlim(-ax_max, ax_max)
    ax.set_ylim(-ax_max, ax_max)
    plt.axvline(x=0, color="black")
    plt.axhline(y=0, color="black")
    plt.legend(loc='upper right', fontsize=FONTSIZE)
    plt.savefig("{0}/{1}.png".format(folder, i), dpi=70)
    plt.close()

    return compressed_point_i


def plot_some_random_compressed_points(X: np.ndarray, all_compressed_point: np.ndarray, filename: str,
                                       ax_max: plt.Axes):
    """Plot a 3x3 grid with the features and one random point that is compressed for each subplot."""
    fig, axis = plt.subplots(3, 3, figsize=(10, 8))
    axis_flat = axis.flat
    sampled_indices = random.sample(range(len(all_compressed_point)), 9)
    for idx in range(len(axis_flat)):
        ax = axis_flat[idx]
        i = sampled_indices[idx]
        ax.scatter(X[:250, 0], X[:250, 1], color=COLORS[0], alpha=0.5, label="No compression")
        ax.scatter(all_compressed_point[i, 0], all_compressed_point[i, 1], color=COLORS[1], label="Compression")
        ax.plot(X[i][0], X[i][1], marker="o", markersize=6, markerfacecolor="red", label="Point of interest")
        ax.set_xlim(-ax_max, ax_max)
        ax.set_ylim(-ax_max, ax_max)
        ax.axvline(x=0, color="black", lw=1)
        ax.axhline(y=0, color="black", lw=1)
    axis_flat[0].legend(loc='best')
    filename += "/grid"
    if USE_ORTHO_MATRIX:
        filename += "-ortho"
    plt.savefig("{0}.pdf".format(filename), bbox_inches='tight', dpi=600)
    plt.close()


def plot_compression_process_by_compressor(dataset: SyntheticDataset, compressor: CompressionModel,
                                           data_covariance: np.ndarray, covariance_compr: np.ndarray,
                                           compressed_points: np.ndarray, ax: plt.Axes):
    """For the given compressor: (1) plot a 3x3 grid with one random compressed point, (2) create a gif showing how
    features are compressed, and (3) add the points scatter plot with their corresponding ellipse to the axis. """

    X = dataset.X
    eigenvalues, eigenvectors = np.linalg.eig(data_covariance)
    if dataset.use_ortho_matrix:
        ax_max = max(eigenvalues) * (2 if WITH_STANDARDISATION else 0.95) #1.5 #* 0.61
    else:
        ax_max = max(eigenvalues) * (2.25 if WITH_STANDARDISATION else 1) #2.25 #* 0.61

    folder = FOLDER + "/" + compressor.get_name()
    create_folder_if_not_existing(folder)

    for i in tqdm(range(10)):
        plot_compressed_points(compressor, X, i, folder, ax_max*1.2)

    compressed_points = compressed_points[:min(SIZE_DATASET, NB_COMPRESSED_POINT)]
    plot_some_random_compressed_points(X, compressed_points, folder, ax_max*1.2)
    add_scatter_plot_to_figure(ax, X, compressed_points, compressor, data_covariance, covariance_compr,
                               ax_max)

    filenames = ["{0}/{1}.png".format(folder, i) for i in range(min(SIZE_DATASET, 10))]
    gif_name = "{0}/{1}-animation{2}".format(folder, compressor.get_name(), '-ortho' if USE_ORTHO_MATRIX else "")
    create_gif(file_names=filenames, gif_name=gif_name + ".gif")

    return ax_max


def plot_compression_process(dataset: SyntheticDataset, my_compressors: List[CompressionModel],
                             covariances: List[np.ndarray], compressed_points: List[np.ndarray], labels: List[str]):
    """For all compressors, build the scatter plot with their corresponding ellipse, create a 3x3 grid with random
    compressed points and a gif."""
    nb_of_columns = len(labels) - 1
    fig_distrib, axes_distrib = plt.subplots(1, nb_of_columns, figsize=(4*nb_of_columns, 4))
    axes_distrib = axes_distrib.flatten()

    for i in range(1, len(my_compressors)):
        compressor = my_compressors[i]
        ax_max = plot_compression_process_by_compressor(dataset, compressor, covariances[0], covariances[i],
                                                        compressed_points[i], axes_distrib[i - 1])

    # We add the yticks only for the first subplot.
    x_ticks = np.array([-0.9, -0.6, -0.3, 0, 0.3, 0.6, 0.9]) * ax_max
    axes_distrib[0].set_yticks(list(x_ticks), labelsize=FONTSIZE)
    axes_distrib[0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.subplots_adjust(wspace=0, hspace=0)

    filename = FOLDER + "scatter_plot"
    if USE_ORTHO_MATRIX:
        filename = "{0}-ortho".format(filename)
    if WITH_STANDARDISATION:
        filename = "{0}-std".format(filename)

    fig_distrib.subplots_adjust(wspace=0, hspace=0)
    fig_distrib.savefig("{0}.pdf".format(filename), bbox_inches='tight', dpi=600)


if __name__ == '__main__':

    synthetic_dataset = SyntheticDataset()
    synthetic_dataset.generate_constants(DIM, SIZE_DATASET, POWER_COV, R_SIGMA, NB_CLIENTS, USE_ORTHO_MATRIX,
                                         eigenvalues=EIGENVALUES, client_id=0, heterogeneity=HETEROGENEITY, w0_seed=None)
    synthetic_dataset.define_compressors()
    synthetic_dataset.generate_X(FEATURES_DISTRIBUTION)

    if WITH_STANDARDISATION:
        synthetic_dataset.normalize()

    labels = ["no compr.", "quantiz.", "stab. quantiz.", "gauss. proj.", "sparsif", "rand1", "all or noth."]
    no_compressor = Quantization(0, dim=DIM)
    synthetic_dataset.rand1 = RandK(1, dim=DIM, biased=False)
    my_compressors = [no_compressor, synthetic_dataset.quantizator,
                      synthetic_dataset.stabilized_quantizator,
                      synthetic_dataset.sketcher,
                      synthetic_dataset.sparsificator,
                      synthetic_dataset.rand1,
                      synthetic_dataset.all_or_nothinger]

    all_covariances, all_compressed_points = get_all_covariances(synthetic_dataset, my_compressors)
    all_theoretical_covariances = [get_theoretical_cov(synthetic_dataset, 1, c.get_name()) for c in my_compressors]
    print("Printing all covariance matrices.")
    for i in range(len(labels)):
        print(labels[i])
        print(all_covariances[i])

    plot_compression_process(dataset=synthetic_dataset, my_compressors=my_compressors, covariances=all_covariances,
                             compressed_points=all_compressed_points, labels=labels)