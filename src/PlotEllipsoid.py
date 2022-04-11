"""Created by Constantin Philippenko, 7th April 2022."""
import math

from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

from PlotUtils import confidence_ellipse
from src.CompressionModel import *
from src.SyntheticDataset import SyntheticDataset
from src.Utilities import create_folder_if_not_existing

SIZE_DATASET = 10**2
DIM = 100
POWER_COV = 4
R_SIGMA=0

DIM = 2

USE_ORTHO_MATRIX = True
DO_LOGISTIC_REGRESSION = False

NB_RANDOM_COMPRESSION = 25

COLORS = ["tab:blue", "tab:orange", "tab:brown", "tab:green", "tab:red", "tab:purple"]

def plot_compressed_points(compressor, X, i, folder, ax_max):
    scatter_plot = np.array([compressor.compress(X[i]) for j in range(NB_RANDOM_COMPRESSION)])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(X[:, 0], X[:, 1], color=COLORS[0], label="No compression")
    ax.scatter(scatter_plot[:, 0], scatter_plot[:, 1], color=COLORS[1], s=80, label="Compression")
    plt.plot(X[i][0], X[i][1], marker="o", markersize=10, markerfacecolor="red", label="Point of interest")
    ax.set_xlim(-ax_max, ax_max)
    ax.set_ylim(-ax_max, ax_max)
    plt.axvline(x=0, color="black")
    plt.axhline(y=0, color="black")
    plt.legend(loc='upper right')
    plt.savefig("{0}/{1}.png".format(folder, i), dpi=70) # default is 100, to reduce how many pixels the figures has.
    plt.close()

    return scatter_plot


def create_gif(file_names, gif_name):
    images = [Image.open(fn) for fn in file_names]
    images[0].save(gif_name, format="GIF", append_images=images,
                   save_all=True, duration=400, loop=0)


def compute_quadratic_error(x, all_compression):
    return np.mean([(c - x)**2 for c in all_compression])


def compute_and_create_gif(dataset, compressor, ax, ax_mse0, ax_mse1):

    X = dataset.X_complete
    ax_max = max([X[i,j] for i in range(X.shape[0]) for j in range(2)]) / dataset.LEVEL_RDK

    folder = "pictures/ellipse/" + compressor.get_name()
    create_folder_if_not_existing(folder)

    scatter_plot = []
    for i in tqdm(range(SIZE_DATASET)):
        scatter_plot.append(plot_compressed_points(compressor, X, i, folder, ax_max))

    MSE_0 = [compute_quadratic_error(X[i][0], scatter_plot[i][:, 0]) for i in range(SIZE_DATASET)]
    MSE_1 = [compute_quadratic_error(X[i][1], scatter_plot[i][:, 1]) for i in range(SIZE_DATASET)]
    norms = [np.linalg.norm(X[i], ord=2) for i in range(SIZE_DATASET)]
    norms, MSE_0 = zip(*sorted(zip(norms, MSE_0)))
    ax_mse0.plot(norms, MSE_0, label = compressor.get_name())
    norms, MSE_1 = zip(*sorted(zip(norms, MSE_1)))
    ax_mse1.plot(norms, MSE_1, label = compressor.get_name())

    scatter_plot = np.concatenate(scatter_plot)


    ax.scatter(X[:, 0], X[:, 1], color=COLORS[0], label="No compression")
    ax.scatter(scatter_plot[:, 0], scatter_plot[:, 1], color=COLORS[1], s=80, label="Compression")
    ax.set_xlim(-ax_max, ax_max)
    ax.set_ylim(-ax_max, ax_max)
    ax.legend(loc='upper right')
    ax.axvline(x=0, color="black")
    ax.axhline(y=0, color="black")
    ax.set_title(compressor.get_name())

    filenames = ["{0}/{1}.png".format(folder, i) for i in range(SIZE_DATASET)]
    create_gif(file_names=filenames, gif_name=folder + ".gif")

def compute_covariance(dataset, compressor):
    X = dataset.X_complete

    X_compressed = X.copy()

    for i in tqdm(range(SIZE_DATASET)):
        X_compressed[i] = compressor.compress(X[i])

    cov_matrix = X_compressed.T.dot(X_compressed) / SIZE_DATASET

    diag = np.diag(cov_matrix)
    return diag, cov_matrix, X_compressed


def get_all_covariances(dataset: SyntheticDataset):

    no_compressor = SQuantization(0, dim=DIM)

    my_compressors = [no_compressor, dataset.quantizator, dataset.stabilized_quantizator,
                      dataset.sparsificator, dataset.rand_sketcher, dataset.all_or_nothinger]

    fig_distrib, axes_distrib = plt.subplots(3, 2, figsize=(15, 9))
    axes_distrib = [axes_distrib[0,0], axes_distrib[0,1], axes_distrib[1,0], axes_distrib[1,1], axes_distrib[2,0], axes_distrib[2,1]]

    fig_MSE, axes_MSE = plt.subplots(1, 2, figsize=(12, 9))

    for i in range(1, len(my_compressors)):
        compressor = my_compressors[i]
        compute_and_create_gif(dataset, compressor, axes_distrib[i - 1], axes_MSE[0], axes_MSE[1])
    axes_MSE[0].legend(loc='upper left')
    axes_MSE[0].set_title("X axis")
    axes_MSE[1].legend(loc='upper left')
    axes_MSE[1].set_title("Y axis")

    fig_distrib.savefig("pictures/ellipse/scatter_plot.eps", format='eps', bbox_inches='tight',dpi=100)
    fig_MSE.savefig("pictures/ellipse/MSE.eps", format='eps', bbox_inches='tight', dpi=100)


    all_covariances = []
    all_compressed_points = []
    for compressor in my_compressors:
        diag, cov_matrix, compressed_points = compute_covariance(dataset, compressor)
        all_covariances.append(cov_matrix)
        all_compressed_points.append(compressed_points)

    labels = ["no compr.", "quantiz.", "stab. quantiz.", "sparsif", "gauss. proj.", "all or noth."]
    return all_covariances, all_compressed_points, labels


def plot_ellipse(dataset, covariances, labels):
    fig, ax = plt.subplots(figsize=(6, 6))
    for i in range(len(covariances)):
        ax.axvline(c='grey', lw=1)
        ax.axhline(c='grey', lw=1)
        confidence_ellipse(covariances[i], labels[i], ax, edgecolor=COLORS[i], zorder=0)

    ax.scatter(dataset.X_complete[:,0], dataset.X_complete[:,1], s=0.5) # TODO : afficher les points aussi sur l'ellipse.
    ax.scatter(0, 0, c='red', s=3)
    ax.set_title("Ellipse")
    ax.legend(fancybox=True, framealpha=0.5)
    ax.axis('equal')

    fig.subplots_adjust(hspace=0.25)
    plt.savefig("pictures/ellipse/ellipse.eps", format='eps', bbox_inches='tight',dpi=100)
    
if __name__ == '__main__':

    synthetic_dataset = SyntheticDataset()
    synthetic_dataset.generate_constants(DIM, SIZE_DATASET, POWER_COV, R_SIGMA, USE_ORTHO_MATRIX)
    synthetic_dataset.define_compressors()
    synthetic_dataset.generate_X()

    all_covariances, all_compressed_points, labels = get_all_covariances(synthetic_dataset)

    plot_ellipse(dataset=synthetic_dataset, covariances=all_covariances, labels=labels)