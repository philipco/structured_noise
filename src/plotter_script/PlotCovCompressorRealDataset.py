"""
Created by Constantin Philippenko, 1st April 2023.

Used to generate the figure in the appendix of the paper which plots the covariance for both quantum and
cifar10, w.o. compression, w. quantization/sparsification uisng two different levels.
"""
from typing import List

import matplotlib
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.CompressionModel import CompressionModel
from src.RealDataset import RealLifeDataset
from src.utilities.Utilities import create_folder_if_not_existing, get_project_root

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

sns.set(font='serif', style='white',
        palette="tab10",
        font_scale=1.2,
        rc={'text.usetex': True, 'pgf.rcfonts': False})

FOLDER = "{0}/pictures/real_dataset".format(get_project_root())

COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:brown", "tab:purple", "tab:cyan"]

FONTSIZE = 16
LINESIZE = 4


def compute_cov(dataset: RealLifeDataset, compressor: CompressionModel, squantization: List[int]) -> List[np.ndarray]:
    """Compute the covariances of the dataset for quantization and sparsification for different level of compression."""

    all_cov = []
    for s in squantization:

        if compressor.get_name() == "Qtzd":
            compressor.reset_level(s)
        if compressor.get_name() == "Sparsification":
            compressor.reset_level(1 / (np.sqrt(dataset.dim) / s + 1))
        X = dataset.X
        X_compressed = dataset.X.copy()

        print("\n>>>>>>> {0}\t omega: {1}".format(compressor.get_name(), compressor.omega_c))
        for i in tqdm(range(dataset.size_dataset)):
            X_compressed[i] = compressor.compress(X[i])

        all_cov.append(np.cov(X_compressed.T))

    return all_cov


cmap = "coolwarm"

if __name__ == '__main__':

    datasets = ["quantum", "cifar10"]

    for dataset_name in datasets:
        print(">>>>> {0}".format(dataset_name))

        dataset = RealLifeDataset(dataset_name)

        fig, axes = plt.subplots(1, 2, figsize=(4 * 2, 4))
        axes[0].imshow(dataset.upper_sigma_raw, cmap=cmap)
        axes[1].imshow(dataset.upper_sigma, cmap=cmap)
        for i in range(len(axes)):
            axes[i].get_xaxis().set_visible(False)
            axes[i].get_yaxis().set_visible(False)
        plt.show()
        folder = "{0}/cov_matrix".format(FOLDER)
        create_folder_if_not_existing(folder)
        plt.savefig("{0}/{1}_no_compr_cov.pdf".format(folder, dataset_name),
                    bbox_inches='tight', dpi=600)

        my_compressors = [dataset.quantizator, dataset.sparsificator]

        squantization = [16, 2] if dataset_name == "cifar10" else [6, 1]

        for compressor in my_compressors:
            list_of_cov = compute_cov(dataset, compressor, squantization)

            fig, axes = plt.subplots(1, len(list_of_cov), figsize=(4 * len(list_of_cov), 4))
            for i in range(len(axes)):
                axes[i].get_xaxis().set_visible(False)
                axes[i].get_yaxis().set_visible(False)
                axes[i].imshow(list_of_cov[i], cmap=cmap)
            folder = "{0}/cov_matrix".format(FOLDER)
            create_folder_if_not_existing(folder)
            plt.savefig("{0}/{1}_{2}_cov.pdf".format(folder, dataset_name, compressor.get_name()),
                        bbox_inches='tight', dpi=600)
