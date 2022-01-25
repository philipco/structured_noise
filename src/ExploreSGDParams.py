"""
Created by Constantin Philippenko, 18th January 2022.
"""
import numpy as np

from src.CompressionModel import SQuantization, RandomSparsification
from HyperparametersExploration import Exploration
from src.SGD import SGD
from src.SyntheticDataset import SyntheticDataset
from Metric import Metric
from src.hyperparameters_exploration.Explorer import Explorer
from src.hyperparameters_exploration.Hyperparameters import Hyperparameters


SIZE_DATASET = 10000000
DIM = 10
POVER_COVARIANCE = 2


def explore_by_sigma(power_cov: int):
    synthetic_dataset = SyntheticDataset()
    synthetic_dataset.generate_dataset(DIM, size_dataset=SIZE_DATASET, power_cov=power_cov, use_ortho_matrix=False)
    sgd = SGD(synthetic_dataset)
    optimal_loss = sgd.compute_true_risk(synthetic_dataset.w_star, synthetic_dataset.X, synthetic_dataset.Y)

    losses, avg_losses, _, _ = sgd.gradient_descent()

    losses_quantization, avg_losses_quantization, _, _ = sgd.gradient_descent_compression(synthetic_dataset.quantizator)
    losses_sparsification, avg_losses_sparsification, _, _ = sgd.gradient_descent_compression(synthetic_dataset.sparsificator)

    return [avg_losses - optimal_loss, avg_losses_quantization - optimal_loss, avg_losses_sparsification - optimal_loss]


def explore_by_omega(omega):
    synthetic_dataset = SyntheticDataset()
    synthetic_dataset.generate_dataset(100, size_dataset=SIZE_DATASET, power_cov=POVER_COVARIANCE, use_ortho_matrix=False)
    sgd = SGD(synthetic_dataset)
    optimal_loss = sgd.compute_true_risk(synthetic_dataset.w_star, synthetic_dataset.X, synthetic_dataset.Y)

    level_qtzt = omega # np.floor(np.sqrt(self.dim) / TARGET_OMEGA)  # Lead to omega_c = 3.
    quantizator = SQuantization(level_qtzt, dim=DIM)

    level_spar = 1 / (quantizator.omega_c + 1)
    sparsificator = RandomSparsification(level_spar, dim=DIM)

    losses_quantization, avg_losses_quantization, w, matrix_qtz = sgd.gradient_descent_compression(quantizator)
    losses_sparsification, avg_losses_sparsification, w, matrix_sparse = sgd.gradient_descent_compression(sparsificator)

    return [avg_losses_quantization - optimal_loss, avg_losses_sparsification - optimal_loss]


def explore_by_dim(dim: int):
    synthetic_dataset = SyntheticDataset()
    synthetic_dataset.generate_dataset(dim, size_dataset=SIZE_DATASET, power_cov=2, use_ortho_matrix=False)
    sgd = SGD(synthetic_dataset)
    optimal_loss = sgd.compute_true_risk(synthetic_dataset.w_star, synthetic_dataset.X, synthetic_dataset.Y)

    losses, avg_losses, _, _ = sgd.gradient_descent()

    losses_quantization, avg_losses_quantization, _, _ = sgd.gradient_descent_compression(synthetic_dataset.quantizator)
    losses_sparsification, avg_losses_sparsification, _, _ = sgd.gradient_descent_compression(synthetic_dataset.sparsificator)

    return [avg_losses - optimal_loss, avg_losses_quantization - optimal_loss, avg_losses_sparsification - optimal_loss]


def loss_average(loss):
    return np.log10(loss[-1])


if __name__ == '__main__':
    metric = Metric("Averaged loss", y_axis_label=r"$\log_{10}(F(\bar w) - F(w_*))$", compute=loss_average)

    explorer = Explorer(["no compression", "quantization", "sparsification"], explore_by_dim)

    hyperparameters = Hyperparameters(range_hyperparameters=[50, 100, 200, 500, 750, 1000],
                                      name=r"Impact of the dimension $d$",
                                      x_axis_label=r"$d \in \mathbb{N}$")

    exploration = Exploration(name="Impact of dim", hyperparameters=hyperparameters, explorer=explorer,
                              metrics=metric)
    exploration.run_exploration()
    exploration.plot_exploration()


    explorer = Explorer(["no compression", "quantization", "sparsification"], explore_by_sigma)

    hyperparameters = Hyperparameters(range_hyperparameters=[1,2,3,4,5],
                                      name=r"Impact of the covariance matric $\Sigma$",
                                      x_axis_label=r"$r \in \mathbb{N}$, s.c. $\Sigma = Diag(1/i^r)_{i=1}^d$")

    exploration = Exploration(name="Impact of sigma", hyperparameters=hyperparameters, explorer=explorer,
                              metrics=metric)
    exploration.run_exploration()
    exploration.plot_exploration()

    # explorer = Explorer(["quantization", "sparsification"], explore_by_omega)
    #
    # hyperparameters = Hyperparameters(range_hyperparameters=[1, 2, 4, 8, 16],
    #                                   name=r"Impact of $\omega_c$",
    #                                   x_axis_label=r"$s \in \mathbb{N}$, the level for quantization")
    #
    # exploration = Exploration(name="Impact of omega", hyperparameters=hyperparameters, explorer=explorer,
    #                           metrics=metric)
    # exploration.run_exploration()
    # exploration.plot_exploration()

