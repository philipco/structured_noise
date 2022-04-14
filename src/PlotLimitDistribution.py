"""Created by Constantin Philippenko, 11th April 2022."""

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm

from src.PlotUtils import confidence_ellipse
from src.CompressionModel import *
from src.SGD import SGDCompressed
from src.SyntheticDataset import SyntheticDataset
from src.Utilities import create_folder_if_not_existing
from src.main import plot_only_avg

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

SIZE_DATASET = 10**4
POWER_COV = 2
R_SIGMA = 0

DIM = 2
EIGENVALUES = np.array([1,10])

USE_ORTHO_MATRIX = True
DO_LOGISTIC_REGRESSION = False

LAST_POINTS = 0
NB_TRY = 50

FOLDER = "pictures/TCL/muL={0}".format(min(EIGENVALUES)/max(EIGENVALUES))
create_folder_if_not_existing(FOLDER)

COLORS = ["tab:blue", "tab:orange", "tab:brown", "tab:green", "tab:red", "tab:purple", "tab:cyan"]


def compute_inversion(matrix):
    return np.linalg.inv(matrix)


def compute_limit_distrib(inv_sigma, error_cov):
    return inv_sigma @ error_cov @ inv_sigma


def compute_empirical_covariance(random_vector):
    return random_vector.T.dot(random_vector) / random_vector.shape[0]


def get_legend_limit_distrib():
    legend_color = [Line2D([0], [0], color=COLORS[0], lw=2, label='no compr.'),
                    Line2D([0], [0], color=COLORS[1], lw=2, label='compr.')]

    legend_line = [Line2D([0], [0], linestyle="--", color="black", lw=1, label=r"$\mathrm{Cov}( \sqrt{n} (\bar w - w_*))$"),
                   Line2D([0], [0], linestyle="-", color="black", lw=1,
                          label=r"$\Sigma^{-1} \mathrm{Cov}(\mathcal C (x)) \Sigma^{-1}$"),
                   Line2D([0], [0], linestyle=":", color="black", lw=1, label=r"$\Sigma^{-1} C \Sigma^{-1}$"),
                   Line2D([], [], color='black', marker="*", linestyle='None',
                          markersize=4, label=r"last $\bar w - w_*$")
                   ]
    return legend_line, legend_color


def plot_limit_distribution(sigma, covariances, avg_dist_to_opt, compressors):

    inv_sigma = compute_inversion(sigma)

    cut_avg_dist_to_opt = [dist_to_opt * np.sqrt(SIZE_DATASET) for dist_to_opt in avg_dist_to_opt]
    cut_covariances = covariances

    fig_distrib, axes_distrib = plt.subplots(2, 3, figsize=(10, 8))
    axes_distrib = axes_distrib.flat
    for i in range(len(covariances) - 1):
        axes_distrib[i].axvline(c='grey', lw=1)
        axes_distrib[i].axhline(c='grey', lw=1)

        axes_distrib[i].plot(cut_avg_dist_to_opt[i + 1][-1, 0], cut_avg_dist_to_opt[i + 1][-1, 1], marker ="*", color=COLORS[1], markersize=10)
        confidence_ellipse(compute_limit_distrib(inv_sigma, cut_covariances[i+1]), "",
                           axes_distrib[i], edgecolor=COLORS[1], zorder=0, lw=2)
        confidence_ellipse(compute_empirical_covariance(cut_avg_dist_to_opt[i + 1]), "", axes_distrib[i], edgecolor=COLORS[1],
                           zorder=0, linestyle="--", lw=2)

        axes_distrib[i].plot(cut_avg_dist_to_opt[0][-1, 0], cut_avg_dist_to_opt[0][-1, 1], marker="*", color=COLORS[0], markersize=10)
        confidence_ellipse(compute_limit_distrib(inv_sigma, cut_covariances[0]), "",
                           axes_distrib[i], edgecolor=COLORS[0], zorder=0, lw=2)
        confidence_ellipse(compute_empirical_covariance(cut_avg_dist_to_opt[0]), "", axes_distrib[i],
                           edgecolor=COLORS[0], zorder=0, linestyle="--", lw=2)

        axes_distrib[i].scatter(0, 0, c='red', s=3)
        axes_distrib[i].set_title(compressors[i+1].get_name())

    legend_line, legend_color = get_legend_limit_distrib()

    first_legend = axes_distrib[0].legend(handles=legend_color, loc='lower right', fancybox=True, framealpha=0.5)

    axes_distrib[0].add_artist(first_legend)
    axes_distrib[0].legend(handles=legend_line, loc='upper right', fancybox=True, framealpha=0.5)
    axes_distrib[0].axis('equal')

    # fig.subplots_adjust(hspace=0.25)
    filename = "pictures/limit_distribution/TCL/5"
    if USE_ORTHO_MATRIX:
        filename = "{0}-ortho".format(filename)
    plt.savefig("{0}.png".format(filename), bbox_inches='tight', dpi=600)


def plot_TCL_without_compression(ax, sigma, empirical_cov, avg_dist_to_opt, title):

    inv_sigma = compute_inversion(sigma)
    compute_limit_distrib(inv_sigma, empirical_cov)

    avg_dist_to_opt *= np.sqrt(SIZE_DATASET)

    confidence_ellipse(compute_empirical_covariance(avg_dist_to_opt), r"$\mathrm{Cov}( \sqrt{n} (\bar w_n - w_*))$", ax,
                       edgecolor=COLORS[1], zorder=0, lw=2)
    confidence_ellipse(compute_limit_distrib(inv_sigma, empirical_cov),
                       r"$\Sigma^{-1} \mathrm{Cov}(\mathcal C (x)) \Sigma^{-1}$", ax, edgecolor=COLORS[1],
                       linestyle="--", zorder=0, lw=2)
    confidence_ellipse(inv_sigma,  r"$\Sigma^{-1}$", ax,
                       edgecolor=COLORS[0], zorder=0, lw=2, linestyle=":")

    ax.scatter(avg_dist_to_opt[:, 0], avg_dist_to_opt[:, 1], color=COLORS[1], alpha=0.5)

    ax.scatter(0, 0, c='red', s=3)
    ax.axvline(c='grey', lw=1)
    ax.axhline(c='grey', lw=1)
    ax.set_title(title)


def plot_TCL(sigma, all_covariances, all_avg_sgd, labels):
    fig, ax = plt.subplots(figsize=(6, 5))
    plot_TCL_without_compression(ax, sigma, all_covariances[0], all_avg_sgd[0], title=labels[0])
    ax.legend(loc='upper right', fancybox=True, framealpha=0.5)
    filename = "{0}/N{1}-TCL_without_compression".format(FOLDER, SIZE_DATASET)
    if USE_ORTHO_MATRIX:
        filename = "{0}-ortho".format(filename)
    plt.savefig("{0}.png".format(filename), bbox_inches='tight', dpi=600)

    fig_TCL, axes_TCL = plt.subplots(2, 3, figsize=(10, 8))
    axes_TCL = axes_TCL.flat
    for idx_compressor in range(1, len(all_covariances)):
        plot_TCL_without_compression(axes_TCL[idx_compressor-1], sigma, all_covariances[idx_compressor],
                                     all_avg_sgd[idx_compressor], title=labels[idx_compressor])
    axes_TCL[0].legend(loc='upper right', fancybox=True, framealpha=0.5)
    filename = "{0}/N{1}-TCL_with_compression".format(FOLDER, SIZE_DATASET)
    if USE_ORTHO_MATRIX:
        filename = "{0}-ortho".format(filename)
    plt.savefig("{0}.png".format(filename), bbox_inches='tight', dpi=600)


def compute_covariance_of_compressors(dataset, compressor):
    X = dataset.X_complete
    X_compressed = X.copy()
    for i in range(X.shape[0]):
        X_compressed[i] = compressor.compress(X[i])
    return compute_empirical_covariance(X_compressed)


def get_all_covariances_of_compressors(dataset: SyntheticDataset, my_compressors):
    all_covariances = []
    for compressor in tqdm(my_compressors):
        cov_matrix = compute_covariance_of_compressors(dataset, compressor)
        all_covariances.append(cov_matrix)

    return all_covariances

if __name__ == '__main__':

    synthetic_dataset = SyntheticDataset()
    synthetic_dataset.generate_dataset(DIM, size_dataset=SIZE_DATASET, power_cov=POWER_COV, r_sigma=R_SIGMA,
                                       use_ortho_matrix=USE_ORTHO_MATRIX, do_logistic_regression=DO_LOGISTIC_REGRESSION,
                                       eigenvalues=EIGENVALUES)

    no_compressor = SQuantization(0, dim=DIM)
    my_compressors = [no_compressor, synthetic_dataset.quantizator, synthetic_dataset.stabilized_quantizator,
                      synthetic_dataset.rand_sketcher,
                      synthetic_dataset.sparsificator, synthetic_dataset.rand1,
                      synthetic_dataset.all_or_nothinger]
    labels = ["No compression"] + [compressor.get_name() for compressor in my_compressors[1:]]

    all_covariances = get_all_covariances_of_compressors(synthetic_dataset, my_compressors)

    all_avg_sgd = [[] for idx_compressor in range(len(my_compressors))]
    all_sgd_descent = []

    for idx in range(NB_TRY):
        ### Very important to regenerate dataset in order to compute the variance of the last avg iterate. ###
        synthetic_dataset.regenerate_dataset()

        for idx_compressor in range(len(my_compressors)):
            compressor = my_compressors[idx_compressor]
            sgd = SGDCompressed(synthetic_dataset, compressor).gradient_descent(label=compressor.get_name(),
                                                                                deacreasing_step_size=True)
            # We save only the excess loss of the first try.
            if idx == 0:
                all_sgd_descent.append(sgd)
            all_avg_sgd[idx_compressor].append(sgd.all_avg_w[-1:] - synthetic_dataset.w_star)

        # We plot the excess loss of the first try.
        if idx == 0:
            plot_only_avg(all_sgd_descent[1:], sgd_nocompr=all_sgd_descent[0], optimal_loss=0,
                          hash_string="TCL/muL={0}/N{1}-avg_sgd".format(min(EIGENVALUES)/max(EIGENVALUES), SIZE_DATASET))

    all_avg_sgd = [np.concatenate(avg_sgd) for avg_sgd in all_avg_sgd]
    plot_TCL(synthetic_dataset.upper_sigma, all_covariances, all_avg_sgd, labels)


