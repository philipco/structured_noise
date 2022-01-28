"""
Created by Constantin Philippenko, 18th January 2022.
"""
import sys

import numpy as np

from src.CompressionModel import SQuantization, RandomSparsification
from src.hyperparameters_exploration.HyperparametersExploration import Exploration
from src.SGD import SGD
from src.SyntheticDataset import SyntheticDataset
from src.hyperparameters_exploration.Metric import Metric
from src.hyperparameters_exploration.Explorer import Explorer
from src.hyperparameters_exploration.Hyperparameters import Hyperparameters


SIZE_DATASET = 10**7
DIM = 100
POWER_COVARIANCE = 4
R_SIGMA = 0


def explore_by_sigma(power_cov: int):
    synthetic_dataset = SyntheticDataset()
    synthetic_dataset.generate_dataset(DIM, size_dataset=SIZE_DATASET, power_cov=power_cov, r_sigma=R_SIGMA,
                                       use_ortho_matrix=False)
    sgd = SGD(synthetic_dataset)
    optimal_loss = sgd.compute_true_risk(synthetic_dataset.w_star, synthetic_dataset.X, synthetic_dataset.Y)

    sgd_nocompr = sgd.gradient_descent()

    sgd_qtz = sgd.gradient_descent_compression(synthetic_dataset.quantizator)
    sgd_rdk = sgd.gradient_descent_compression(synthetic_dataset.sparsificator)

    return [sgd_nocompr.avg_losses - optimal_loss, sgd_qtz.avg_losses - optimal_loss, sgd_rdk.avg_losses - optimal_loss]


def explore_by_wstar(r_sigma: int):
    synthetic_dataset = SyntheticDataset()
    synthetic_dataset.generate_dataset(DIM, size_dataset=SIZE_DATASET, power_cov=POWER_COVARIANCE, r_sigma=r_sigma,
                                       use_ortho_matrix=False)
    sgd = SGD(synthetic_dataset)
    optimal_loss = sgd.compute_true_risk(synthetic_dataset.w_star, synthetic_dataset.X, synthetic_dataset.Y)

    sgd_nocompr = sgd.gradient_descent()

    sgd_qtz = sgd.gradient_descent_compression(synthetic_dataset.quantizator)
    sgd_rdk = sgd.gradient_descent_compression(synthetic_dataset.sparsificator)

    return [sgd_nocompr.avg_losses - optimal_loss, sgd_qtz.avg_losses - optimal_loss, sgd_rdk.avg_losses - optimal_loss]


def explore_by_level_qtz(level_qtzt):
    synthetic_dataset = SyntheticDataset()
    synthetic_dataset.generate_dataset(dim=DIM, size_dataset=SIZE_DATASET, power_cov=POWER_COVARIANCE, r_sigma=R_SIGMA,
                                       use_ortho_matrix=False)
    sgd = SGD(synthetic_dataset)
    optimal_loss = sgd.compute_true_risk(synthetic_dataset.w_star, synthetic_dataset.X, synthetic_dataset.Y)

    sgd_nocompr = sgd.gradient_descent()

    quantizator = SQuantization(level_qtzt, dim=DIM)
    level_rdk = 1 / (quantizator.omega_c + 1)
    sparsificator = RandomSparsification(level_rdk, dim=DIM)

    sgd_qtz = sgd.gradient_descent_compression(quantizator)
    sgd_rdk = sgd.gradient_descent_compression(sparsificator)

    return [sgd_nocompr.avg_losses - optimal_loss, sgd_qtz.avg_losses - optimal_loss, sgd_rdk.avg_losses - optimal_loss]


def explore_by_dim(dim: int):
    synthetic_dataset = SyntheticDataset()
    synthetic_dataset.generate_dataset(dim, size_dataset=SIZE_DATASET, power_cov=POWER_COVARIANCE, r_sigma=R_SIGMA,
                                       use_ortho_matrix=False)
    sgd = SGD(synthetic_dataset)
    optimal_loss = sgd.compute_true_risk(synthetic_dataset.w_star, synthetic_dataset.X, synthetic_dataset.Y)

    sgd_nocompr = sgd.gradient_descent()

    sgd_qtz = sgd.gradient_descent_compression(synthetic_dataset.quantizator)
    sgd_rdk = sgd.gradient_descent_compression(synthetic_dataset.sparsificator)

    return [sgd_nocompr.avg_losses - optimal_loss, sgd_qtz.avg_losses - optimal_loss, sgd_rdk.avg_losses - optimal_loss]


def explore_by_gamma(gamma_factor: int):
    synthetic_dataset = SyntheticDataset()
    synthetic_dataset.generate_dataset(DIM, size_dataset=SIZE_DATASET, power_cov=POWER_COVARIANCE, r_sigma=R_SIGMA,
                                       use_ortho_matrix=False)
    synthetic_dataset.gamma = gamma_factor * synthetic_dataset.gamma
    sgd = SGD(synthetic_dataset)
    optimal_loss = sgd.compute_true_risk(synthetic_dataset.w_star, synthetic_dataset.X, synthetic_dataset.Y)

    sgd_nocompr = sgd.gradient_descent()

    sgd_qtz = sgd.gradient_descent_compression(synthetic_dataset.quantizator)
    sgd_rdk = sgd.gradient_descent_compression(synthetic_dataset.sparsificator)

    return [sgd_nocompr.avg_losses - optimal_loss, sgd_qtz.avg_losses - optimal_loss, sgd_rdk.avg_losses - optimal_loss]


def loss_average(loss):
    return np.log10(loss[-1])


if __name__ == '__main__':
    metric = Metric("Averaged loss", y_axis_label=r"$\log_{10}(F(\bar w) - F(w_*))$", compute=loss_average)

    explorer = Explorer(["no compression", "quantization", "sparsification"], explore_by_wstar)

    if sys.argv[1] == "wstar":
        hyperparameters = Hyperparameters(range_hyperparameters=[1/8, 1/4, 1/2, 0, 1, 2],
                                          name=r"Impact of the true model $w_*$",
                                          x_axis_label="$r \in \mathbb{N}$, s.c. $w_* = \Sigma^r Vect(1)$")

        exploration = Exploration(name="Impact of dim", hyperparameters=hyperparameters, explorer=explorer,
                                  metrics=metric)
        exploration.run_exploration()
        exploration.plot_exploration()

    if sys.argv[1] == "dim":
        explorer = Explorer(["no compression", "quantization", "sparsification"], explore_by_dim)

        hyperparameters = Hyperparameters(range_hyperparameters=[10, 100, 200, 500, 750, 1000],
                                          name=r"Impact of the dimension $d$",
                                          x_axis_label="$d \in \mathbb{N}$")

        exploration = Exploration(name="Impact of dim", hyperparameters=hyperparameters, explorer=explorer,
                                  metrics=metric)
        exploration.run_exploration()
        exploration.plot_exploration()

    if sys.argv[1] == "cov":
        explorer = Explorer(outputs_label=["no compression", "quantization", "sparsification"],
                            function=explore_by_sigma)

        hyperparameters = Hyperparameters(range_hyperparameters=[1,2,3,4,5],
                                          name=r"Impact of the covariance matric $\Sigma$",
                                          x_axis_label=r"$r \in \mathbb{N}$, s.c. $\Sigma = Diag(1/i^r)_{i=1}^d$")

        exploration = Exploration(name="Impact of sigma", hyperparameters=hyperparameters, explorer=explorer,
                                  metrics=metric)
        exploration.run_exploration()
        exploration.plot_exploration()

    if sys.argv[1] == "gamma":
        explorer = Explorer(outputs_label=["no compression", "quantization", "sparsification"],
                            function=explore_by_gamma)

        hyperparameters = Hyperparameters(range_hyperparameters=[0.25,0.5,1,2,4,6],
                                          name=r"Impact of the step size $\gamma$",
                                          x_axis_label=r"$\rho \in \mathbb{N}$, s.c. $\gamma = \frac{\rho}{L ( 1 + 2 (\omega_c + 1))}$")

        exploration = Exploration(name="Impact of sigma", hyperparameters=hyperparameters, explorer=explorer,
                                  metrics=metric)
        exploration.run_exploration()
        exploration.plot_exploration()

    if sys.argv[1] == "level_qtz":
        explorer = Explorer(outputs_label=["no compression", "quantization", "sparsification"],
                            function=explore_by_level_qtz)

        hyperparameters = Hyperparameters(range_hyperparameters=[1, 2, 4, 8, 16, 0],
                                          name=r"Impact of $\omega_c$",
                                          x_axis_label=r"$s \in \mathbb{N}$, the level for quantization")

        exploration = Exploration(name="Impact of omega", hyperparameters=hyperparameters, explorer=explorer,
                                  metrics=metric)
        exploration.run_exploration()
        exploration.plot_exploration()

