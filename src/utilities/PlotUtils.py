"""Created by Constantin Philippenko, 7th April 2022."""
import math
from typing import List

import numpy as np
from PIL import Image
from matplotlib import transforms, pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.ticker import FormatStrFormatter

from src.CompressionModel import CompressionModel
from src.SGD import SeriesOfSGD

FONTSIZE = 15
LINESIZE = 3

COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
COLORS_DOUBLE = ["tab:blue", "tab:orange", "tab:orange", "tab:green", "tab:green", "tab:red", "tab:red",
          "tab:purple", "tab:purple", "tab:brown", "tab:brown"]


def create_gif(file_names: List[str], gif_name: str, duration: int = 400, loop: int = 0):
    images = [Image.open(fn) for fn in file_names]
    images[0].save(gif_name, format="GIF", append_images=images,
                   save_all=True, duration=duration, loop=loop)


def plot_eigen_values(all_sgd: SeriesOfSGD, hash_string: str = None, custom_legend: List = None):
    fig, ax = plt.subplots(figsize=(6.5, 6))

    i = 0
    for label, list_of_sgd in all_sgd.dict_of_sgd.items():
        dim = list_of_sgd[0].dim

        diag_cov = np.mean([np.log10(sgd.diag_cov_gradients) for sgd in list_of_sgd], axis=0)
        plt.plot(np.log10(np.arange(1, dim + 1)), diag_cov, label=label, lw=LINESIZE)
        i+=1
    ax.tick_params(axis='both', labelsize=15)

    l1 = ax.legend(loc='lower left', fontsize=FONTSIZE)
    if custom_legend is not None:
        l2 = ax.legend(handles=custom_legend, loc="upper right", fontsize=FONTSIZE)
        ax.add_artist(l2)
    ax.add_artist(l1)

    ax.set_xlabel(r"$\log(i), \forall i \in \{1, ..., d\}$", fontsize=15)
    ax.set_ylabel(r"$\log(Diag(\frac{\mathcal C (X)^T.\mathcal C (X)}{n})_i)$", fontsize=15)
    if hash_string:
        plt.savefig('{0}-eigenvalues.pdf'.format("./pictures/" + hash_string), bbox_inches='tight', dpi=600)
        plt.close()
    else:
        plt.show()


def setup_plot_with_SGD(all_sgd: SeriesOfSGD, optimal_loss: float, hash_string: str = None, custom_legend: List = None,
                        with_artemis: bool = False):
    fig, axes = plt.subplots(2, figsize=(8, 7))

    i = 0
    for label, list_of_sgd in all_sgd.dict_of_sgd.items():

        losses = np.mean([np.log10(sgd.losses - optimal_loss) for sgd in list_of_sgd], axis=0)

        avg_losses = np.mean([np.log10(sgd.avg_losses - optimal_loss) for sgd in list_of_sgd], axis=0)
        avg_losses_var = np.std([np.log10(sgd.avg_losses - optimal_loss) for sgd in list_of_sgd], axis=0)
        log_xaxis = np.log10(list_of_sgd[0].log_xaxis)

        label_sgd = None if "-art" in label else "SGD {0}".format(label)
        label_avg_sgd = None if "-art" in label else "AvgSGD {0}".format(label)
        line_style = "--" if "-art" in label else "-"
        if with_artemis:
            color = COLORS[(i-1) // 2  + 1] if i > 0 else COLORS[0]
        else:
            color = COLORS[i]
        axes[0].plot(log_xaxis, losses, label=label_sgd, lw=LINESIZE, linestyle=line_style, color=color)

        axes[1].plot(log_xaxis, avg_losses, label=label_avg_sgd, lw=LINESIZE, linestyle=line_style, color=color)

        axes[1].fill_between(log_xaxis, avg_losses - avg_losses_var, avg_losses + avg_losses_var, alpha=0.2)
        i+=1

    for ax in axes:
        l1 = ax.legend(loc='lower left', fontsize=FONTSIZE)
        if custom_legend is not None:
            l2 = ax.legend(handles=custom_legend, loc="upper right")
            ax.add_artist(l2)
        ax.add_artist(l1)
        ax.grid(True)
    axes[0].set_ylabel(r"$\log_{10}(F(w_k) - F(w_*))$", fontsize=15)
    axes[1].set_ylabel(r"$\log_{10}(F(\overline{w}_k) - F(w_*))$", fontsize=15)
    axes[1].set_xlabel(r"$\log_{10}(k)$", fontsize=15)

    if hash_string:
        plt.savefig('{0}.pdf'.format("./pictures/" + hash_string), bbox_inches='tight', dpi=600)
        plt.close()
    else:
        plt.show()


def plot_only_avg(all_sgd: SeriesOfSGD, optimal_loss: float, hash_string: str = None, custom_legend: List = None,
                  with_artemis: bool = False):
    fig, ax = plt.subplots(figsize=(8, 4))

    i = 0
    for label, list_of_sgd in all_sgd.dict_of_sgd.items():
        avg_losses = np.mean([np.log10(sgd.avg_losses - optimal_loss) for sgd in list_of_sgd], axis=0)
        avg_losses_var = np.std([np.log10(sgd.avg_losses - optimal_loss) for sgd in list_of_sgd], axis=0)
        log_xaxis = np.log10(list_of_sgd[0].log_xaxis)
        label_avg_sgd = None if "-art" in label else "{0}".format(label)
        line_style = "--" if "-art" in label else "-"
        if with_artemis:
            color = COLORS[(i - 1) // 2 + 1] if i > 0 else COLORS[0]
        else:
            color = COLORS[i]
        ax.plot(log_xaxis, avg_losses, label=label_avg_sgd, lw=LINESIZE, linestyle=line_style, color=color)
        plt.fill_between(log_xaxis, avg_losses - avg_losses_var, avg_losses + avg_losses_var, alpha=0.2, color=color)
        i+=1

    l1 = ax.legend(loc='lower left', fontsize=FONTSIZE)
    if custom_legend is not None:
        l2 = ax.legend(handles=custom_legend, loc="upper right")
        ax.add_artist(l2)
    ax.add_artist(l1)

    ax.grid(True)
    ax.set_ylabel(r"$\log_{10}(F(\overline{w}_k) - F(w_*))$", fontsize=FONTSIZE)
    ax.set_xlabel(r"$\log_{10}(k)$", fontsize=FONTSIZE)

    if hash_string:
        plt.savefig('{0}.pdf'.format("./pictures/" + hash_string), bbox_inches='tight', dpi=600)
        plt.close()
    else:
        plt.show()


def add_scatter_plot_to_figure(ax: plt.Axes, X: np.ndarray, all_compressed_point: np.ndarray,
                               compressor: CompressionModel, data_covariance:np.ndarray, covariance: np.ndarray,
                               ax_max: int, plot_eig: bool = True, nb_pts_to_plot: int = 250, taille_pts: int = 10):

    mask = (np.abs(X[:, 0]) <= ax_max) & (np.abs(X[:, 1]) <= ax_max)
    result = X[mask]
    ax.scatter(result[:min(nb_pts_to_plot, len(result)), 0], result[:min(nb_pts_to_plot, len(result)), 1],
               color=COLORS[0], alpha=0.5, s=taille_pts, label="No compression", zorder=2)

    mask = (np.abs(all_compressed_point[:, 0]) <= ax_max) & (np.abs(all_compressed_point[:, 1]) <= ax_max)
    result = all_compressed_point[mask]
    ax.scatter(result[:min(nb_pts_to_plot, len(result)), 0], result[:min(nb_pts_to_plot, len(result)), 1],
               color=COLORS[1], alpha=0.5, s=taille_pts + 5, label="Compression", zorder=1)

    plot_ellipse(data_covariance, r"$\mathcal{E}_{\mathrm{Cov}[{x_k}}]$", ax, color=COLORS[0], linestyle="-", zorder=3,
                 lw=LINESIZE, n_std=4, plot_eig=plot_eig)

    plot_ellipse(covariance, r"$\mathcal{E}_{\mathrm{Cov}[{\mathcal{C} (x_k)}}]$", ax, color=COLORS[1],
                 linestyle="-", zorder=3, lw=LINESIZE, n_std=4, plot_eig=plot_eig)

    ax.set_title(compressor.get_name(), fontsize=FONTSIZE)

    ax.axis('equal')
    if ax_max is not None:
        ax.set_xlim(-ax_max, ax_max)
        ax.set_ylim(-ax_max, ax_max)
        x_ticks = np.array([-0.8, -0.4, 0, 0.4, 0.8]) * ax_max
        ax.set_xticks(list(x_ticks))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.set_yticks([]) # We remove the yticks for all subplots.

    ax.axvline(x=0, color="black")
    ax.axhline(y=0, color="black")



def plot_ellipse(cov: np.ndarray, label: str, ax: plt.Axes, n_std: float = 1.0, plot_eig: bool = False, **kwargs):
    """ Plot an ellipse centered on zero given a 2D-covariance matrix."""

    Q, D, _ = np.linalg.svd(cov)

    size = 5000
    x = np.linspace(-math.sqrt(n_std * D[0]), math.sqrt(n_std * D[0]), size)
    y1 = np.sqrt(D[1] * (n_std - x**2/D[0]))
    y2 = - np.sqrt(D[1] * (n_std - x**2/D[0]))
    y = np.concatenate([y1, np.flip(y2)])
    x = np.concatenate([x, np.flip(x)])
    X = Q[0,0] * x + Q[0,1] * y
    Y = Q[1,0] * x + Q[1,1] * y

    ax.plot(X, Y, label = label, **kwargs)

    if kwargs["color"] == COLORS[0]:
        kwargs["zorder"] = 5
    else:
        kwargs["zorder"] = 4
    kwargs["lw"] = 4
    if plot_eig:
        V1 = 0.9 * Q @ np.array([math.sqrt(n_std * D[0]), 0])
        V2 = 0.9 * Q @ np.array([0, math.sqrt(n_std * D[1])])
        ax.arrow(0, 0, V1[0], V1[1], head_width=0.5, **kwargs)
        ax.arrow(0, 0, V2[0], V2[1], head_width=0.5, **kwargs)


def confidence_ellipse(cov: np.ndarray, label: str, ax: plt.Axes, n_std: float = 1.0, facecolor: str = 'none',
                       **kwargs):
    """
    Plot an ellipse centered on zero given a 2D-covariance matrix.

    Parameters
    ----------
    cov : array-like, shape (n, n)
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y)

    ellipse.set_transform(transf + ax.transData)

    ellipse.set(clip_box=ax.bbox, label=label)
    patch = ax.add_patch(ellipse)
    return patch, max(scale_x, scale_y)