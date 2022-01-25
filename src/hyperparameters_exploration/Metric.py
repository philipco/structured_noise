"""
Created by Constantin Philippenko, 20th January 2022.
"""

class Metric:

    def __init__(self, name, y_axis_label, compute):
        self.name = name
        self.y_axis_label = y_axis_label
        self.compute = compute
