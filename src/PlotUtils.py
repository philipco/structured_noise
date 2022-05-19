"""Created by Constantin Philippenko, 7th April 2022."""
import numpy as np
from PIL import Image
from matplotlib import transforms, pyplot as plt
from matplotlib.patches import Ellipse

from src.SGD import SGDRun


def create_gif(file_names, gif_name, duration: int = 400, loop: int = 0):
    images = [Image.open(fn) for fn in file_names]
    images[0].save(gif_name, format="GIF", append_images=images,
                   save_all=True, duration=duration, loop=loop)


def plot_SGD_and_AVG(axes, sgd_run: SGDRun, optimal_loss):

    axes[0].plot(np.arange(len(sgd_run.losses)), np.log10(sgd_run.losses - optimal_loss),
                 label="SGD {0}".format(sgd_run.label))
    axes[1].plot(np.arange(len(sgd_run.losses)), np.log10(sgd_run.avg_losses - optimal_loss),
                 label="AvgSGD {0}".format(sgd_run.label))


def setup_plot_with_SGD(all_sgd, sgd_nocompr: SGDRun, optimal_loss, hash_string: str = None):
    fig, axes = plt.subplots(2, figsize=(8, 7))

    plot_SGD_and_AVG(axes, sgd_nocompr, optimal_loss)

    for sgd_try in all_sgd:
        plot_SGD_and_AVG(axes, sgd_try, optimal_loss)

    for ax in axes:
        ax.legend(loc='lower left', fontsize=10)
        # ax.set_ylim(top=0.5)
        ax.grid(True)
    axes[0].set_ylabel(r"$\log_{10}(F(w_k) - F(w_*))$", fontsize=15)
    axes[1].set_ylabel(r"$\log_{10}(F(\bar w_k) - F(w_*))$", fontsize=15)
    axes[1].set_xlabel(r"$\log_{10}(n)$", fontsize=15)

    if hash_string:
        plt.savefig('{0}.eps'.format("./pictures/" + hash_string), format='eps')
        plt.close()
    else:
        plt.show()

def plot_only_avg(all_sgd, sgd_nocompr: SGDRun, optimal_loss, hash_string: str = None):
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(np.log10(sgd_nocompr.log_xaxis), np.log10(sgd_nocompr.avg_losses - optimal_loss),
                 label="{0}".format(sgd_nocompr.label))

    for sgd_try in all_sgd:
        ax.plot(np.log10(sgd_try.log_xaxis), np.log10(sgd_try.avg_losses - optimal_loss),
                label="{0}".format(sgd_try.label))

    ax.legend(loc='best', fontsize=15)
    ax.set_ylim(top=0.5)
    ax.grid(True)
    ax.set_ylabel(r"$\log_{10}(F(\bar w_k) - F(w_*))$", fontsize=15)
    ax.set_xlabel(r"$\log_{10}(n)$", fontsize=15)
    ax.set_title("Avg SGD")

    if hash_string:
        plt.savefig('{0}.png'.format("./pictures/" + hash_string), bbox_inches='tight', dpi=600)
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