"""Created by Constantin Philippenko, 7th April 2022."""
from typing import List

import numpy as np
from PIL import Image
from matplotlib import transforms, pyplot as plt
from matplotlib.legend import Legend
from matplotlib.patches import Ellipse

from src.SGD import SGDRun

FONTSIZE = 17
LINESIZE = 3

COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
COLORS_DOUBLE = ["tab:blue", "tab:orange", "tab:orange", "tab:green", "tab:green", "tab:red", "tab:red",
          "tab:purple", "tab:purple", "tab:brown", "tab:brown"]

def create_gif(file_names, gif_name, duration: int = 400, loop: int = 0):
    images = [Image.open(fn) for fn in file_names]
    images[0].save(gif_name, format="GIF", append_images=images,
                   save_all=True, duration=duration, loop=loop)


def plot_eigen_values(all_sgd, hash_string: str = None, custom_legend: List = None):
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


def setup_plot_with_SGD(all_sgd, optimal_loss, hash_string: str = None, custom_legend: List = None, with_artemis=False):
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
            color = COLORS[i // 2 + 1] if "-art" in label else COLORS[
                i // 2 + 1]  # index shift because we must exclude the blue colors (for vanilla SGD).
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
        # ax.set_ylim(top=0.5)
        ax.grid(True)
    axes[0].set_ylabel(r"$\log_{10}(F(w_k) - F(w_*))$", fontsize=15)
    axes[1].set_ylabel(r"$\log_{10}(F(\overline{w}_k) - F(w_*))$", fontsize=15)
    axes[1].set_xlabel(r"$\log_{10}(k)$", fontsize=15)

    if hash_string:
        plt.savefig('{0}.pdf'.format("./pictures/" + hash_string), bbox_inches='tight', dpi=600)
        plt.close()
    else:
        plt.show()


def plot_only_avg(all_sgd, optimal_loss, hash_string: str = None, custom_legend: List = None, with_artemis=False):
    fig, ax = plt.subplots(figsize=(8, 4))

    i = 0
    for label, list_of_sgd in all_sgd.dict_of_sgd.items():
        avg_losses = np.mean([np.log10(sgd.avg_losses - optimal_loss) for sgd in list_of_sgd], axis=0)
        avg_losses_var = np.std([np.log10(sgd.avg_losses - optimal_loss) for sgd in list_of_sgd], axis=0)
        log_xaxis = np.log10(list_of_sgd[0].log_xaxis)
        label_avg_sgd = None if "-art" in label else "{0}".format(label)
        line_style = "--" if "-art" in label else "-"
        if with_artemis:
            color = COLORS[i//2+1] if "-art" in label else COLORS[i//2+1] # index shift because we must exclude the blue colors (for vanilla SGD).
        else:
            color = COLORS[i]
        ax.plot(log_xaxis, avg_losses, label=label_avg_sgd, lw=LINESIZE, linestyle=line_style, color=color)
        plt.fill_between(log_xaxis, avg_losses - avg_losses_var, avg_losses + avg_losses_var, alpha=0.2)
        i+=1

    l1 = ax.legend(loc='lower left', fontsize=FONTSIZE)
    if custom_legend is not None:
        l2 = ax.legend(handles=custom_legend, loc="upper right")
        ax.add_artist(l2)
    ax.add_artist(l1)

    # ax.set_ylim(top=0.5)
    ax.grid(True)
    ax.set_ylabel(r"$\log_{10}(F(\overline{w}_k) - F(w_*))$", fontsize=FONTSIZE)
    ax.set_xlabel(r"$\log_{10}(k)$", fontsize=FONTSIZE)
    # ax.set_title("Avg SGD")

    if hash_string:
        plt.savefig('{0}.pdf'.format("./pictures/" + hash_string), bbox_inches='tight', dpi=600)
        plt.close()
    else:
        plt.show()


def plot_ellipse(cov, label, ax, n_std=1.0, plot_eig: bool = False, **kwargs):
    """ Plot an ellipse centered on zero given a 2D-covariance matrix."""
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    length_axis = np.sqrt(eigenvalues)

    size = 1000
    theta = np.linspace(0, 2 * np.pi, size)

    ellipse = np.array([n_std * length_axis[0] * np.cos(theta), n_std * length_axis[1] * np.sin(theta)])
    rotated_ellipse = np.zeros((2, size))
    for i in range(size):
        rotated_ellipse[:, i] = eigenvectors @ ellipse[:, i]

    ax.plot(rotated_ellipse[0, :], rotated_ellipse[1, :], label = label, **kwargs)

    if plot_eig:
        origin = np.array([[0, 0], [0, 0]])
        ax.quiver(*origin, length_axis * eigenvectors[0, :], length_axis * eigenvectors[1, :],
                  angles='xy', scale_units='xy', scale=1, **kwargs)

    return np.max(rotated_ellipse)


def confidence_ellipse(cov, label, ax, n_std=1.0, facecolor='none', **kwargs):
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
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
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