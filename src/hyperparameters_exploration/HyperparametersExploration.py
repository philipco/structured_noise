"""
Created by Constantin Philippenko, 18th January 2022.
"""
import matplotlib
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

import hashlib
import os
import sys

import numpy as np
from matplotlib import pyplot as plt

from src.PickleHandler import pickle_saver
from src.Utilities import create_folder_if_not_existing
from src.hyperparameters_exploration import Explorer
from src.hyperparameters_exploration.Hyperparameters import Hyperparameters
from src.hyperparameters_exploration.Metric import Metric


class Exploration:

    def __init__(self, name, hyperparameters: Hyperparameters, explorer: Explorer, metrics: Metric):
        # super().__init__()
        self.name = name
        self.hyperparameters = hyperparameters
        self.explorer = explorer
        self.metrics = metrics
        self.nb_runs = 2
        self.results = np.zeros((self.explorer.nb_outputs, self.nb_runs, self.hyperparameters.nb_hyperparams))
        self.string_before_hash = str(self.hyperparameters.range_hyperparameters) + self.explorer.function.__name__
        self.hash_string = hashlib.shake_256(self.string_before_hash.encode()).hexdigest(4) # returns a hash value of length 2*4
        self.pickle_folder = "./pickle/"
        self.pictures_folder = "./pictures/exploration/"
        create_folder_if_not_existing(self.pickle_folder)
        create_folder_if_not_existing(self.pictures_folder)


    def run_exploration(self):
        print("====> Starting exploration : ", self.name)
        for idx_param in range(self.hyperparameters.nb_hyperparams):
            param = self.hyperparameters.range_hyperparameters[idx_param]
            print("Hyperparameter's value:", param)
            # self.blockPrint()
            for idx_run in range(self.nb_runs):
                output = self.explorer.explore(param)
                for i in range(len(output)):
                    self.results[i, idx_run, idx_param] = self.metrics.compute(output[i])
                    pickle_saver(self, self.pickle_folder + self.string_before_hash)
            self.enablePrint()

    def plot_exploration(self):
        fig, ax = plt.subplots(figsize=(8, 7))

        for i in range(len(self.explorer.outputs_label)):
            plt.errorbar(range(self.hyperparameters.nb_hyperparams), np.mean(self.results[i], axis=0),
                         yerr=np.std(self.results[i], axis=0),
                         label=self.explorer.outputs_label[i],
                         lw=4)
        plt.xticks([i for i in range(0, len(self.hyperparameters.range_hyperparameters))],
                   self.hyperparameters.range_hyperparameters,
                   rotation=30, fontsize=15)
        plt.yticks(fontsize=15)
        ax.set_xlabel(self.hyperparameters.x_axis_label, fontsize=15)
        ax.set_ylabel(self.metrics.y_axis_label, fontsize=15)
        plt.title(self.hyperparameters.name, fontsize=15)
        plt.legend(loc='best', fontsize=15)
        ax.grid()
        plt.savefig('{0}.eps'.format(self.pictures_folder + self.hash_string), format='eps')
        plt.close()

    # Disable
    def blockPrint(self):
        sys.stdout = open(os.devnull, 'w')

    # Restore
    def enablePrint(self):
        sys.stdout = sys.__stdout__


