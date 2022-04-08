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

NB_RANDOM_COMPRESSION = 50

COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

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
    plt.savefig("{0}/{1}.png".format(folder, i))#, format='eps')
    plt.close()

    return scatter_plot


def create_gif(file_names, gif_name):
    images = [Image.open(fn) for fn in file_names]
    images[0].save(gif_name, format="GIF", append_images=images,
                   save_all=True, duration=150, loop=0)


def compute_and_create_gif(dataset, compressor):
    X = dataset.X_complete
    # ax_max = math.sqrt(max([X[i].T @ X[i] for i in range(X.shape[0])])) / dataset.LEVEL_RDK
    ax_max = max([X[i,j] for i in range(X.shape[0]) for j in range(2)]) / dataset.LEVEL_RDK

    folder = "pictures/ellipse/" + compressor.get_name()
    create_folder_if_not_existing(folder)

    scatter_plot = []
    for i in tqdm(range(SIZE_DATASET)):
        scatter_plot.append(plot_compressed_points(compressor, X, i, folder, ax_max))

    scatter_plot = np.concatenate(scatter_plot)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(X[:, 0], X[:, 1], color=COLORS[0], label="No compression")
    ax.scatter(scatter_plot[:, 0], scatter_plot[:, 1], color=COLORS[1], s=80, label="Compression")
    ax.set_xlim(-ax_max, ax_max)
    ax.set_ylim(-ax_max, ax_max)
    plt.legend(loc='upper right')
    plt.axvline(x=0, color="black")
    plt.axhline(y=0, color="black")
    plt.savefig("{0}-scatter_plot.eps".format(folder), format='eps')

    filenames = ["{0}/{1}.png".format(folder, i) for i in range(SIZE_DATASET)]
    create_gif(file_names=filenames, gif_name=folder + ".gif")

def compute_covariance(dataset, compressor):
    X = dataset.X_complete

    X_compressed = X.copy()

    for i in tqdm(range(SIZE_DATASET)):
        X_compressed[i] = compressor.compress(X[i])

    cov_matrix = X_compressed.T.dot(X_compressed) / SIZE_DATASET

    # if USE_ORTHO_MATRIX:
    #     cov_matrix = dataset.ortho_matrix.T.dot(cov_matrix).dot(dataset.ortho_matrix)

    diag = np.diag(cov_matrix)
    return diag, cov_matrix, X_compressed


def get_all_covariances(dataset: SyntheticDataset):

    no_compressor = SQuantization(0, dim=DIM)

    my_compressors = [no_compressor, dataset.quantizator, dataset.sparsificator, dataset.rand_sketcher,
                      dataset.all_or_nothinger]

    for compressor in my_compressors[1:]:
        compute_and_create_gif(dataset, compressor)

    all_covariances = []
    all_compressed_points = []
    for compressor in my_compressors:
        diag, cov_matrix, compressed_points = compute_covariance(dataset, compressor)
        all_covariances.append(cov_matrix)
        all_compressed_points.append(compressed_points)

    labels = ["no compr.", "quantiz.", "rdk", "gauss. proj.", "all or noth."]
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
    plt.savefig("pictures/ellipse/ellipse.eps", format='eps')
    
if __name__ == '__main__':

    synthetic_dataset = SyntheticDataset()
    synthetic_dataset.generate_constants(DIM, SIZE_DATASET, POWER_COV, R_SIGMA, USE_ORTHO_MATRIX)
    synthetic_dataset.define_compressors()
    synthetic_dataset.generate_X()

    all_covariances, all_compressed_points, labels = get_all_covariances(synthetic_dataset)

    for i in range(len(all_covariances)):
        print(labels[i])
        print(all_covariances[i])

    plot_ellipse(dataset=synthetic_dataset, covariances=all_covariances, labels=labels)