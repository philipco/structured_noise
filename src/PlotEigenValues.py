"""
Created by Constantin Philippenko, 17th January 2022.
"""
import cmath

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.CompressionModel import SQuantization, RandomSparsification, Sketching
from src.SyntheticDataset import SyntheticDataset
from src.Utilities import create_folder_if_not_existing

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

SIZE_DATASET = 10**5
DIM = 50
POWER_COV = 4
R_SIGMA=0

USE_ORTHO_MATRIX = False


def prepare_sparsification(x, p):
    rademacher = np.random.binomial(1, 0.5, size=len(x))
    rademacher[rademacher == 0] = -1
    return x * (rademacher) # * cmath.sqrt(p-1))


def compute_diag(dataset, compressor):

    X = dataset.X_complete

    X_compressed = X.copy()
    for i in tqdm(range(SIZE_DATASET)):
        X_compressed[i] = compressor.compress(X[i])

    cov_matrix = X_compressed.T.dot(X_compressed) / SIZE_DATASET

    if USE_ORTHO_MATRIX:
        cov_matrix = dataset.ortho_matrix.T.dot(cov_matrix).dot(dataset.ortho_matrix)

    diag = np.diag(cov_matrix)
    return diag, cov_matrix

def compute_diag_matrices(dim: int):

    dataset = SyntheticDataset()
    dataset.generate_constants(dim, size_dataset=SIZE_DATASET, power_cov=POWER_COV, r_sigma=R_SIGMA,
                       use_ortho_matrix=USE_ORTHO_MATRIX)
    dataset.define_compressors()
    dataset.generate_X()

    no_compressor = SQuantization(0, dim=dim)

    quantizator = SQuantization(1, dim=dim)

    p = 1 / (quantizator.omega_c + 1)
    print("Sparsification proba: ", p)
    sparsificator = RandomSparsification(p, dim, biased=False)

    gaussian_sketcher = Sketching(p, dim, type_proj = "gaussian")
    rand_gaussian_sketcher = Sketching(p, dim, randomized=True, type_proj = "gaussian")

    sparse_sketcher = Sketching(p, dim, type_proj="gaussian")
    rand_sparse_sketcher = Sketching(p, dim, randomized=True, type_proj="gaussian")

    my_compressors = [no_compressor, quantizator, sparsificator, gaussian_sketcher, rand_gaussian_sketcher,
                      sparse_sketcher, rand_sparse_sketcher]

    labels = ["no compr.", "quantiz.", "rdk", "gauss. proj.", "rd gauss. proj.", "sparse proj.", "rd sparse proj."]

    all_diagonals = []
    for compressor in my_compressors:
        diag, cov_matrix = compute_diag(dataset, compressor)
        all_diagonals.append(diag)

    return all_diagonals, labels, dataset.string_for_hash()


if __name__ == '__main__':

    all_diagonals, labels, hash_dataset = compute_diag_matrices(dim=DIM)

    # plt.imshow(cov_matrix)
    # plt.colorbar()
    # plt.title("No compression", fontsize=15)
    # plt.show()
    #
    # plt.imshow(cov_matrix_qtz)
    # plt.colorbar()
    # plt.title("Quantization", fontsize=15)
    # plt.show()
    #
    # plt.imshow(cov_matrix_sparse)
    # plt.colorbar()
    # plt.title("Sparsification", fontsize=15)
    # plt.show()

    fig, ax = plt.subplots(figsize=(6.5, 6))
    for (diagonal, label) in zip(all_diagonals, labels):
        plt.plot(np.log10(np.arange(1, DIM + 1)), np.log10(diagonal), label=label, lw = 2)
    ax.tick_params(axis='both', labelsize=15)
    ax.legend(loc='best', fontsize=15)
    ax.set_xlabel(r"$\log(i), \forall i \in \{1, ..., d\}$", fontsize=15)
    ax.set_ylabel(r"$\log(Diag(\frac{\mathcal C (X)^T.\mathcal C (X)}{n})_i)$", fontsize=15)
    plt.legend(loc='best', fontsize=15)
    folder = "pictures/epsilon_eigenvalues/"
    create_folder_if_not_existing(folder)
    plt.savefig("{0}/{1}.eps".format(folder, hash_dataset), format='eps')

    plt.show()

