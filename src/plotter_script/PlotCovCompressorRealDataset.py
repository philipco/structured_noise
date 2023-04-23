"""
Created by Constantin Philippenko, 17th January 2022.
"""
import copy

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm

from src.CompressionModel import Quantization
from src.utilities.PickleHandler import pickle_saver
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

NB_CLIENTS = 1
# DATASET_NAME = "Flowers102" # TODO Food101 : 1h
OMEGA = 2

BAR_PLOT = False

# In the case of heterogeneoux sigma and with non-diag H, check that random state of the orthogonal matrix is set to 5.
USE_ORTHO_MATRIX = False
HETEROGENEITY = "homog"

FONTSIZE = 16
LINESIZE = 4


def compute_cov(dataset: RealLifeDataset, compressor, squantization):


    all_cov = []
    for s in squantization:

        if compressor.get_name() == "Qtzd":
            compressor.reset_level(s)
        if compressor.get_name() == "Sparsification":
            compressor.reset_level(1/ (np.sqrt(dataset.dim) / s + 1))
        X = dataset.X_complete
        X_compressed = dataset.X_complete.copy()


        print("\n>>>>>>> {0}\t omega: {1}".format(compressor.get_name(), compressor.omega_c))
        for i in tqdm(range(dataset.size_dataset)):
            X_compressed[i] = compressor.compress(X[i])

        all_cov.append(np.cov(X_compressed.T))

    return all_cov


if __name__ == '__main__':

    datasets = ["quantum", "cifar10"]

    for dataset_name in datasets:
        print(">>>>> {0}".format(dataset_name))

        dataset = RealLifeDataset(dataset_name)
        my_compressors = [dataset.quantizator, dataset.sparsificator]

        squantization = [16, 2] if dataset_name == "cifar10" else [6, 1]

        for compressor in my_compressors:
            list_of_cov = compute_cov(dataset, compressor, squantization)

            fig, axes = plt.subplots(1, len(list_of_cov), figsize=(4*len(list_of_cov), 4))
            for i in range(len(axes)):
                axes[i].get_xaxis().set_visible(False)
                axes[i].get_yaxis().set_visible(False)
                axes[i].imshow(list_of_cov[i])
            folder = "{0}/cov_matrix".format(FOLDER)
            create_folder_if_not_existing(folder)
            plt.savefig("{0}/{1}_{2}_cov.pdf".format(folder, dataset_name, compressor.get_name()),
                        bbox_inches='tight', dpi=600)


