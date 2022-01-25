"""
Created by Constantin Philippenko, 20th January 2022.
"""


class Explorer:

    def __init__(self, outputs_label, function):
        self.outputs_label = outputs_label
        self.nb_outputs = len(self.outputs_label)
        self.function = function

    def explore(self, hyperparameter):
        return self.function(hyperparameter)
