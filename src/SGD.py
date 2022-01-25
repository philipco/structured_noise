"""
Created by Constantin Philippenko, 10th January 2022.
"""

import copy

import numpy as np
from tqdm import tqdm

from src.CompressionModel import CompressionModel

DISABLE = True


class SGD():
    NB_EPOCH = 1
    DISABLE = True
    
    def __init__(self, synthetic_dataset) -> None:
        super().__init__()
        self.do_logistic_regression = False
        self.synthetic_dataset = synthetic_dataset
        self.X, self.Y = self.synthetic_dataset.X, self.synthetic_dataset.Y
        self.w_star = self.synthetic_dataset.w_star
        self.GAMMA = self.synthetic_dataset.GAMMA
        self.SIZE_DATASET, self.DIM = self.synthetic_dataset.size_dataset, self.synthetic_dataset.dim
        self.w0 = np.random.normal(0, 1, size = self.DIM)
        self.additive_stochastic_gradient = False

    def compute_empirical_risk(self, w, data, labels):
        if self.do_logistic_regression:
            return  -np.sum(np.log(np.sigmoid(labels * data @ w))) / len(labels)
        return 0.5 * np.linalg.norm(data @ w - labels) ** 2 / len(labels)

    def compute_true_risk(self, w, data, labels):
        if self.do_logistic_regression:
            return  -np.sum(np.log(np.sigmoid(labels * data @ w))) / len(labels)
        return 0.5 * (w - self.w_star).T @ self.synthetic_dataset.upper_sigma @ (w - self.w_star)

    def compute_stochastic_gradient(self, w, data, labels, index):
        x, y = data[index], labels[index]
        if self.do_logistic_regression:
            s = np.sigmoid(y * x @ w)
            return x.T.mv((s - 1) * y) / len(labels)
        return np.array((x @ w - y)).dot(x)

    def compute_additive_stochastic_gradient(self, w, data, labels, index):
        x, y = data[index], labels[index]
        return self.synthetic_dataset.upper_sigma.dot(w) - y * x

    def sgd_update(self, w, gradient, gamma):
        return w - gamma * gradient

    def gradient_descent(self):
        current_w = self.w0
        avg_w = copy.deepcopy(current_w)
        it = 1
        losses = [self.compute_empirical_risk(current_w, self.X, self.Y)]
        avg_losses = [self.compute_empirical_risk(avg_w, self.X, self.Y)]
        matrix_grad = self.X.copy()
        for epoch in range(self.NB_EPOCH):
            indices = np.arange(self.SIZE_DATASET)
            for idx in tqdm(indices, disable=DISABLE):
                gamma = self.synthetic_dataset.GAMMA
                it += 1
                if self.additive_stochastic_gradient:
                    g = self.compute_additive_stochastic_gradient(current_w, self.X, self.Y, idx)
                else:
                    g = self.compute_stochastic_gradient(current_w, self.X, self.Y, idx)
                matrix_grad[idx] = g
                current_w = self.sgd_update(current_w, g, gamma)
                avg_w = current_w / it + avg_w * (it - 1) / it
                losses.append(self.compute_true_risk(current_w, self.X, self.Y))
                avg_losses.append(self.compute_true_risk(avg_w, self.X, self.Y))
        matrix_cov = matrix_grad.T.dot(matrix_grad) / self.SIZE_DATASET
        return losses, avg_losses, current_w, matrix_cov

    def gradient_descent_noised(self):
        current_w = self.w0
        avg_w = copy.deepcopy(current_w)
        it = 1
        losses = [self.compute_empirical_risk(current_w, self.X, self.Y)]
        avg_losses = [self.compute_empirical_risk(avg_w, self.X, self.Y)]
        matrix_grad = self.X.copy()
        for epoch in range(self.NB_EPOCH):
            indices = np.arange(self.SIZE_DATASET)
            for idx in tqdm(indices):
                it += 1
                g = self.compute_stochastic_gradient(current_w, self.X, self.Y, idx) + np.random.normal(0, 1, size = self.DIM)
                matrix_grad[idx] = g
                current_w = self.sgd_update(current_w, g, self.GAMMA)
                avg_w = current_w / it + avg_w * (it - 1) / it
                losses.append(self.compute_true_risk(current_w, self.X, self.Y))
                avg_losses.append(self.compute_true_risk(avg_w, self.X, self.Y))
        matrix_cov = matrix_grad.T.dot(matrix_grad) / self.SIZE_DATASET
        return losses, avg_losses, current_w, matrix_cov


    def gradient_descent_compression(self, compressor: CompressionModel):
        current_w = self.w0
        avg_w = copy.deepcopy(current_w)
        it = 1
        losses = [self.compute_empirical_risk(current_w, self.X, self.Y)]
        avg_losses = [self.compute_empirical_risk(avg_w, self.X, self.Y)]
        matrix_grad = self.X.copy()
        for epoch in range(self.NB_EPOCH):
            indices = np.arange(self.SIZE_DATASET)
            for idx in tqdm(indices):
                gamma = self.synthetic_dataset.GAMMA
                it += 1
                if self.additive_stochastic_gradient:
                    grad = self.compute_additive_stochastic_gradient(current_w, self.X, self.Y, idx)
                else:
                    grad = self.compute_stochastic_gradient(current_w, self.X, self.Y, idx)
                g = compressor.compress(grad)
                matrix_grad[idx] = g
                current_w = self.sgd_update(current_w, g, gamma)
                avg_w = current_w / it + avg_w * (it - 1) / it
                losses.append(self.compute_true_risk(current_w, self.X, self.Y))
                avg_losses.append(self.compute_true_risk(avg_w, self.X, self.Y))

        matrix_cov = matrix_grad.T.dot(matrix_grad) / self.SIZE_DATASET
        return losses, avg_losses, current_w, matrix_cov
