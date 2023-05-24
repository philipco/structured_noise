"""
Created by Constantin Philippenko, 20th December 2021.
"""
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from sympy.physics.control.control_plots import np
from tqdm import tqdm

from src.SyntheticDataset import SyntheticDataset
from src.TheoreticalCov import compute_empirical_covariance, compress_and_compute_covariance
from src.utilities.PlotUtils import add_scatter_plot_to_figure, COLORS
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

FONTSIZE = 15
LINESIZE = 3

USE_ORTHO_MATRIX = True
HETEROGENEITY = "homog"

NB_RANDOM_COMPRESSION = 5
NB_COMPRESSED_POINT = 200

EIGENVALUES = np.array([0.5,0.5])
FOLDER = "pictures/schema/"
create_folder_if_not_existing(FOLDER)


def plot_distribution_and_ellipse(dataset: SyntheticDataset, compressor, ax):

    all_compressed_point = []
    covariance_compr, compressed_points = compress_and_compute_covariance(dataset, compressor)
    data_covariance = compute_empirical_covariance(dataset.X_complete)
    for i in tqdm(range(min(SIZE_DATASET, NB_COMPRESSED_POINT))):
        all_compressed_point.append(np.array([compressor.compress(dataset.X_complete[i]) for j in range(NB_RANDOM_COMPRESSION)]))
    all_compressed_point = np.concatenate(all_compressed_point)

    add_scatter_plot_to_figure(ax, dataset.X_complete, all_compressed_point, compressor, data_covariance, covariance_compr,
                               None, 1.75)

if __name__ == '__main__':

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    axes = axes.flatten()

    synthetic_dataset = SyntheticDataset()
    synthetic_dataset.generate_constants(DIM, SIZE_DATASET, POWER_COV, R_SIGMA, NB_CLIENTS, USE_ORTHO_MATRIX,
                                         eigenvalues=EIGENVALUES, client_id=0, heterogeneity=HETEROGENEITY,
                                         w0_seed=None)
    synthetic_dataset.define_compressors()

    synthetic_dataset.generate_X("normal")
    plot_distribution_and_ellipse(synthetic_dataset, synthetic_dataset.quantizator, axes[0])
    axes[0].set_title("Normal distribution", fontsize=FONTSIZE)

    synthetic_dataset.generate_X("diamond")
    plot_distribution_and_ellipse(synthetic_dataset, synthetic_dataset.quantizator, axes[1])
    axes[1].set_title("Diamond distribution", fontsize=FONTSIZE)

    points_legend = [Line2D([], [], color=COLORS[0], alpha=0.5, marker=".", linestyle='None', markersize=10,
                            label='No compression'),
                     Line2D([], [], color=COLORS[1], alpha=0.5, marker=".", linestyle='None', markersize=10,
                            label='Compression')]
    l1 = axes[1].legend(handles=points_legend, loc='upper left', fontsize=10)

    ellipse_legend = [Line2D([0], [0], color=COLORS[0], lw=2, label=r"$\mathfrak{C}(\mathcal{C}_{\emptyset}, p_{I_2/2})$"),
                     Line2D([0], [0], color=COLORS[1], lw=2, label=r"$\mathfrak{C}(\mathcal{C}_{\mathrm{qtzt}}, p_{I_2/2})$")]
    l2 = axes[1].legend(handles=ellipse_legend, loc="lower right", fontsize=10)
    axes[1].add_artist(l2)
    axes[1].add_artist(l1)

    filename = FOLDER + "normal_vs_diamond"
    fig.savefig("{0}.pdf".format(filename), bbox_inches='tight', dpi=600)