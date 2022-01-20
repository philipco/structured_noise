"""
Created by Constantin Philippenko, 18th January 2022.
"""
import os
import sys

import numpy as np
from matplotlib import pyplot as plt

from hyperparameters_exploration import Explorer
from hyperparameters_exploration.Hyperparameters import Hyperparameters
from hyperparameters_exploration.Metric import Metric


class Exploration:

    def __init__(self, name, hyperparameters: Hyperparameters, explorer: Explorer, metrics: Metric):
        # super().__init__()
        self.name = name
        self.hyperparameters = hyperparameters
        self.explorer = explorer
        self.metrics = metrics
        self.nb_runs = 3
        self.results = np.zeros((self.explorer.nb_outputs, self.nb_runs, self.hyperparameters.nb_hyperparams))

    def run_exploration(self):
        print("====> Starting exploration : ", self.name)
        self.blockPrint()
        for idx_param in range(self.hyperparameters.nb_hyperparams):
            param = self.hyperparameters.range_hyperparameters[idx_param]
            for idx_run in range(self.nb_runs):
                output = self.explorer.explore(param)
                for i in range(len(output)):
                    self.results[i, idx_run, idx_param] = self.metrics.compute(output[i])
        self.enablePrint()

    def plot_exploration(self):
        fig, ax = plt.subplots(figsize=(8, 7))

        for i in range(len(self.explorer.outputs_label)):
            plt.errorbar(range(self.hyperparameters.nb_hyperparams), np.mean(self.results[i], axis=0),
                         yerr=np.std(self.results[i], axis=0),
                         label=self.explorer.outputs_label[i])
        plt.xticks([i for i in range(0, len(self.hyperparameters.range_hyperparameters))],
                   self.hyperparameters.range_hyperparameters,
                   rotation=40, fontsize=15)
        ax.set_xlabel(self.hyperparameters.x_axis_label, fontsize=15)
        ax.set_ylabel(self.metrics.y_axis_label, fontsize=15)
        plt.title(self.hyperparameters.name)
        plt.legend(loc='best', fontsize=15)
        plt.show()

    # Disable
    def blockPrint(self):
        sys.stdout = open(os.devnull, 'w')

    # Restore
    def enablePrint(self):
        sys.stdout = sys.__stdout__


