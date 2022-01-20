"""
Created by Constantin Philippenko, 18th January 2022.
"""
from matplotlib import pyplot as plt

from hyperparameters_exploration.Hyperparameters import Hyperparameters
from hyperparameters_exploration.Metric import Metric


class Exploration:

    def __init__(self, name, hyperparameters: Hyperparameters, lambda_fn, metrics: Metric):
        # super().__init__()
        self.name = name
        self.hyperparameters = hyperparameters
        self.lambda_fn = lambda_fn
        self.metrics = metrics
        self.res = []

    def run_exploration(self):
        for param in self.hyperparameters.range_hyperparameters:
            out = self.lambda_fn(param)
            self.res.append(self.metrics.compute(out))

    def plot_exploration(self):
        fig, ax = plt.subplots(figsize=(8, 7))


        plt.plot(self.res)
        plt.xticks([i for i in range(0, len(self.hyperparameters.range_hyperparameters))],
                   self.hyperparameters.range_hyperparameters,
                   rotation=40, fontsize=15)
        ax.set_xlabel(self.hyperparameters.x_axis_label, fontsize=15)
        ax.set_ylabel(self.metrics.y_axis_label, fontsize=15)
        plt.title(self.hyperparameters.name)
        plt.legend(loc='best', fontsize=15)
        plt.show()


