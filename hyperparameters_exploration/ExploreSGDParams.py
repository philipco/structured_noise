"""
Created by Constantin Philippenko, 18th January 2022.
"""
import numpy as np

from HyperparametersExploration import Exploration
from SGD import SGD
from SyntheticDataset import SyntheticDataset
from Metric import Metric
from hyperparameters_exploration.Hyperparameters import Hyperparameters



SIZE_DATASET = 1000
DIM = 50


def explore_by_sigma(power_cov: int):
    synthetic_dataset = SyntheticDataset()
    synthetic_dataset.generate_dataset(DIM, size_dataset=SIZE_DATASET, power_cov=power_cov, use_ortho_matrix=False)
    sgd = SGD(synthetic_dataset)
    optimal_loss = sgd.compute_true_risk(synthetic_dataset.w_star, synthetic_dataset.X, synthetic_dataset.Y)

    losses_random_sparsification, avg_losses_random_sparsification, w, matrix_sparse = \
        sgd.gradient_descent_compression(synthetic_dataset.sparsificator)

    return avg_losses_random_sparsification - optimal_loss


def loss_average(loss):
    return np.log10(loss[-1])

if __name__ == '__main__':

    hyperparameters = Hyperparameters(range_hyperparameters=[0,1,2,3,4],
                                      name=r"Impact of the covariance matric $\Sigma$",
                                      x_axis_label=r"$r \in \mathbb{N}$, s.c. $\Sigma = Diag(1/i^r)_{i=1}^d$")
    metric = Metric("Averaged loss", y_axis_label=r"$\log_{10}(F(\bar w) - F(w_*))$", compute=loss_average)
    exploration = Exploration(name="Sigma", hyperparameters=hyperparameters, lambda_fn=explore_by_sigma,
                              metrics=metric)
    exploration.run_exploration()
    exploration.plot_exploration()

