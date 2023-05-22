import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from scipy.stats import multivariate_t, multivariate_normal

from src.utilities.PlotUtils import confidence_ellipse
from src.utilities.Utilities import create_folder_if_not_existing

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

SIZE_DATASET = 1000
POWER_COV = 4
R_SIGMA=0

DIM = 2
NB_CLIENTS = 1

FONTSIZE = 22
LINESIZE = 3


FOLDER = "../pictures/schema/"
create_folder_if_not_existing(FOLDER)

COLORS = ["tab:blue", "tab:orange", "tab:brown", "tab:green", "tab:red", "tab:purple", "tab:cyan"]

def plot_ellipse(covariances, law, text_position):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axvline(c='black', lw=1)
    ax.axhline(c='black', lw=1)

    confidence_ellipse(covariances[0], "COV1", ax, edgecolor="tab:red", zorder=0, lw=3)
    plt.text(text_position[0][0], text_position[0][1], r'$\mathrm{Cov}(a)$', horizontalalignment='center',
             verticalalignment='center', transform=ax.transAxes, fontsize=FONTSIZE, color="tab:red")

    confidence_ellipse(covariances[1], "COV2", ax, edgecolor="tab:red", zorder=0, lw=3)
    plt.text(text_position[1][0], text_position[1][1], r'$\mathrm{Cov}(b)$', horizontalalignment='center',
             verticalalignment='center', transform=ax.transAxes, fontsize=FONTSIZE, color="tab:red")

    confidence_ellipse(covariances[2], "COV3", ax, edgecolor="tab:gray", zorder=0, lw=3)

    cov3_label = r'$=\mathrm{Cov}(\frac{a+b}{2})$' if law=="normal" else r'$\mathrm{Cov}(\frac{a+b}{2})$'
    plt.text(text_position[2][0], text_position[2][1], cov3_label, horizontalalignment='center',
             verticalalignment='center', transform=ax.transAxes, fontsize=FONTSIZE, color="tab:gray")

    confidence_ellipse(covariances[3], "AVERAGE", ax, edgecolor="tab:green", zorder=0, lw=5)
    plt.text(text_position[3][0], text_position[3][1], s=r'$\frac{1}{2}(\mathrm{Cov}(a) + \mathrm{Cov}(b)$', horizontalalignment='center',
             verticalalignment='center', transform=ax.transAxes, fontsize=FONTSIZE, color="tab:green")

    ax.scatter(0, 0, c='black', s=5)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    filename = FOLDER + "ellipse_average_" + law
    plt.savefig("{0}.pdf".format(filename), bbox_inches='tight', dpi=600)

def average_of_ellipse():
    law = "student"
    SEED = 10

    Sigma1 = np.array([[1, 0], [0, 0.01]])
    Sigma2 = np.array([[0.01, 0], [0, 1]])
    if law == "student":
        X1 = multivariate_t.rvs(np.zeros(DIM), Sigma1, size=SIZE_DATASET, random_state=SEED, df=1)
        X2 = multivariate_t.rvs(np.zeros(DIM), Sigma2, size=SIZE_DATASET, random_state=SEED, df=1)
        X3 = multivariate_t.rvs(np.zeros(DIM), 0.5 * (Sigma1 + Sigma2), size=SIZE_DATASET, random_state=SEED, df=1)
        text_position = [[0.62, 0.9], [0.91, 0.58], [0.88, 0.08], [0.25, 0.37]]
    elif law == "synth":
        X1 = np.concatenate([np.array([[np.random.normal(0,0.1),1] for k in range(SIZE_DATASET // 2)]),
                             np.array([[np.random.normal(0,0.1),-1] for k in range(SIZE_DATASET // 2)])])
        X2 = np.concatenate([np.array([[1,np.random.normal(0,0.1)] for k in range(SIZE_DATASET // 2)]), np.array([[-1,0] for k in range(SIZE_DATASET // 2)])])
        X3 = np.concatenate([np.array([[1/2, np.random.normal(0,0.1)] for k in range(SIZE_DATASET // 4)]),
                             np.array([[-1/2, 0] for k in range(SIZE_DATASET // 4)]),
                             np.array([[0, -1/2] for k in range(SIZE_DATASET // 4)]),
                             np.array([[0, 1/2] for k in range(SIZE_DATASET // 4)])])
        text_position = [[0.58, 0.9], [0.9, 0.58], [0.62, 0.3], [0.84, 0.18]]
    elif law == "normal":
        X1 = multivariate_normal.rvs(np.zeros(DIM), Sigma1, size=SIZE_DATASET, random_state=SEED)
        X2 = multivariate_normal.rvs(np.zeros(DIM), Sigma2, size=SIZE_DATASET, random_state=SEED)
        X3 = multivariate_normal.rvs(np.zeros(DIM), 0.5 * (Sigma1 + Sigma2), size=SIZE_DATASET, random_state=SEED)
        text_position = [[0.62, 0.9], [0.91, 0.58], [0.25, 0.05], [0.25, 0.12]]
    else:
        raise ValueError("Unknown distribution.")
    X_average = np.concatenate([X1, X2])

    cov1 = X1.T.dot(X1) / len(X1)
    cov2 = X2.T.dot(X2) / len(X2)
    cov3 = X3.T.dot(X3) / len(X3)
    cov_average = X_average.T.dot(X_average) / len(X_average)

    plot_ellipse([cov1, cov2, cov3, cov_average], law, text_position)
    return 0

if __name__ == '__main__':
    average_of_ellipse()