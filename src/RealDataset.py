import copy
import random
from typing import List

import numpy as np
import torchvision.transforms as transforms
from k_means_constrained import KMeansConstrained
from numpy.random import multivariate_normal, dirichlet
from sklearn import decomposition
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, normalize
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.datasets import EMNIST

from src.CompressionModel import RandK
from src.PytorchDatasetClass import QuantumDataset
from src.SyntheticDataset import AbstractDataset
from src.utilities.PickleHandler import pickle_loader, pickle_saver
from src.utilities.Utilities import file_exist, create_folder_if_not_existing, get_path_to_datasets

IMG_SIZE = 16
resize = transforms.Resize((IMG_SIZE, IMG_SIZE))


class RealLifeDataset(AbstractDataset):

    def __init__(self, dataset_name: str, s=None) -> None:
        super().__init__(dataset_name)
        if file_exist("pickle/real_dataset/{0}.pkl".format(dataset_name)):
            myinstance = pickle_loader("pickle/real_dataset/{0}".format(dataset_name))
            for k in myinstance.__dict__.keys():
                setattr(self, k, getattr(myinstance, k))
            self.define_compressors(s)
        else:
            print("Preparing the dataset.")
            self.load_data(dataset_name)
            self.define_compressors(s)
            self.define_w0()
            self.w_star = None
            self.real_dataset = True
            create_folder_if_not_existing("pickle/real_dataset/")
            pickle_saver(self, "pickle/real_dataset/{0}".format(dataset_name))
            print("Done dataset preparation.")

    def define_w0(self, w0_seed: int = 42) -> None:
        """Define the initial point for SGD."""
        if w0_seed is not None:
            self.w0 = np.zeros(self.dim)
        else:
            self.w0 = multivariate_normal(np.zeros(self.dim), np.identity(self.dim) /self.dim)

    def define_compressors(self, s: int = None) -> None:
        super().define_compressors(s=s)
        self.rand1 = RandK(self.sketcher.sub_dim, dim=self.dim, biased=False)
        print("New omega rand1:", self.rand1.omega_c)

    def regenerate_dataset(self) -> None:
        """"Regenerate the dataset for a new SGD."""
        # Concatenate X_complete and Y along the last axis
        data = np.concatenate((self.X, self.Y.reshape(-1, 1)), axis=-1)

        # Shuffle the data
        np.random.shuffle(data)

        # Split X_complete and Y again
        self.X = data[:, :-1]
        self.Y = data[:, -1].astype(np.int64)


    def load_data(self, dataset_name: str):
        """Load the dataset and pre-process it."""
        path_to_dataset = get_path_to_datasets()
        if dataset_name == 'cifar10':

            transform_train = transforms.Compose([
                transforms.ToTensor(),
                resize,
                transforms.Grayscale(),
            ])

            train_data = datasets.CIFAR10(root=path_to_dataset, train=True, download=True, transform=transform_train)

        elif dataset_name == 'cifar100':

            transform_train = transforms.Compose([
                transforms.ToTensor(),
                resize,
                transforms.Grayscale(),
            ])

            train_data = datasets.CIFAR100(root=path_to_dataset, train=True, download=True, transform=transform_train)

        elif dataset_name == 'mnist':
            # CropCenter is required because all pictures have a black band around it, which causes to have eigenvalues
            # equal to zero.
            transform = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop((20,20)), resize,  #transforms.Normalize((0.1307,), (0.3081,))
                                            ])
            train_data = datasets.MNIST(root=path_to_dataset, train=True, download=False, transform=transform)

        elif dataset_name == "fashion-mnist":

            train_transforms = transforms.Compose([transforms.CenterCrop((20,20)), resize, transforms.ToTensor(),  #transforms.Normalize((0.1307,), (0.3081,))
                                                   ])
            train_data = datasets.FashionMNIST(path_to_dataset, download=True, train=True, transform=train_transforms)

        elif dataset_name == "emnist":
            transform = transforms.Compose([transforms.CenterCrop((20,20)), resize, transforms.ToTensor()])
            train_data = EMNIST("./dataset", train=True, download=True, transform=transform, split="balanced")

        elif dataset_name == "quantum":
            train_data = QuantumDataset()


        else:
            raise ValueError("{0} unknown".format(dataset_name))

        batch_size = 128
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

        flat_data = []
        flat_Y = []
        # Iterate over the transformed MNIST training set and print the shape of each image
        for batch_idx, (data, target) in enumerate(train_loader):
            flat_data.append(np.reshape(data.numpy(), (len(data), -1)))
            if len(target.numpy().shape) == 2:
                flat_Y.append(np.concatenate(target.numpy()))
            else:
                flat_Y.append(target.numpy())

        flat_data = np.concatenate(flat_data)
        self.Y = np.concatenate(flat_Y)
        self.dim = len(flat_data[0])
        self.size_dataset = len(flat_data)

        # RAW DATA
        self.X_raw = flat_data

        # DATA WITH NORMALIZATION
        normalize_data = normalize(flat_data)
        self.X_normalized = normalize_data
        print("Mean of the norm:", np.mean([np.linalg.norm(x) for x in self.X_normalized]))

        # DATA WITH STANDARDIZATION
        standardize_data = StandardScaler().fit_transform(flat_data)
        self.X = standardize_data
        print("Mean:", np.mean(standardize_data))
        print("Standard deviation:", np.std(standardize_data))

        # DATA WITH PCA
        pca = decomposition.PCA(self.dim)
        self.X_pca = pca.fit_transform(standardize_data)

        self.upper_sigma = np.cov(self.X.T)
        self.upper_sigma_inv = np.linalg.inv(self.upper_sigma)
        eig, eigvectors = np.linalg.eig(self.upper_sigma)
        big_eig = (eig.real > 10 ** -14).sum()
        eig = np.sort(eig)[::-1]
        if self.dim >= 64:
            print("Eigenvalues - biggest: {0}, 32: {1}, 64: {2}, smallest: {3}".format(eig[0], eig[31], eig[63],
                                                                                       eig[big_eig - 1]))
        else:
            print("Eigenvalues - biggest: {0}, smallest: {1}".format(eig[0], eig[big_eig - 1]))
        print("Warning: there is {0} eigenvalues that are smaller than 10^-14".format(self.dim - big_eig))

        print("Effective dimension:", len(self.X[0]))

        self.trace = np.trace(self.upper_sigma)
        print("Trace H:", self.trace)
        print("Trace H^{-1}:", np.trace(self.upper_sigma_inv))
        print("Computed cov.")

        self.L = eig[0]  # biggest eigenvalues
        print("L=", self.L)

        self.compute_sigma()
        self.compute_sigma_inv()

    def compute_sigma(self) -> None:
        """Compute the covariance of the features without preprocessing, with PCA, with normalization."""
        self.upper_sigma = np.cov(self.X.T)
        self.upper_sigma_pca = np.cov(self.X_pca.T)
        self.upper_sigma_normalized = np.cov(self.X_normalized.T)
        self.upper_sigma_raw = np.cov(self.X_raw.T)



    def compute_sigma_inv(self) -> None:
        """Inverse the covariance of the features without preprocessing, with PCA, with normalization."""
        self.upper_sigma_inv_raw = np.linalg.inv(self.upper_sigma_raw)
        self.upper_sigma_inv = np.linalg.inv(self.upper_sigma)
        self.upper_sigma_inv_pca = np.linalg.inv(self.upper_sigma_pca)
        self.upper_sigma_inv_normalized = np.linalg.inv(self.upper_sigma_normalized)


    def keep_sub_set(self, indices) -> None:
        """Keep a subset of a dataset given some indices."""
        self.size_dataset = len(indices)
        self.X, self.X_pca, self.X_normalized, self.X_raw = self.X[indices], self.X_pca[indices], self.X_normalized[indices], self.X_raw[indices]
        self.Y = self.Y[indices]
        self.compute_sigma()

    def string_for_hash(self, nb_runs: int, stochastic: bool = False, batch_size: int = 1, reg: int = None,
                        step: str = None, heterogeneity: str = None, memory: bool = False) -> str:
        """Return the hash of the dataset."""
        hash = "{0}runs-N{1}-D{2}".format(nb_runs, self.size_dataset, self.dim)
        if self.name:
            hash = "{0}-{1}".format(self.name, hash)
        if not stochastic:
            hash = "{0}-full".format(hash)
        elif batch_size != 1:
            hash = "{0}-b{1}".format(hash, batch_size)
        if reg:
            hash = "{0}-reg{1}".format(hash, reg)
        if step:
            hash = "{0}-{1}".format(hash, step)
        if heterogeneity:
            hash = "{0}-{1}".format(hash, heterogeneity)
        if memory:
            hash = "{0}-mem".format(hash)
        return hash


def diriclet_split(Y: np.ndarray, nb_clients: int, dirichlet_coef: int = 1) -> np.ndarray:
    """Splits the training data by target values (leads to a highly non-iid data distribution) using a Dirichlet
        distribution."""
    unique_values = {}
    targets = Y
    n = len(targets)

    for i in range(n):
        if targets[i] in unique_values:
            unique_values[targets[i]] = np.append(unique_values[targets[i]], [i])
        else:
            unique_values[targets[i]] = np.array([i])
    nb_labels = len(unique_values)

    nb_points_by_clients = n // nb_clients
    matrix = (dirichlet([dirichlet_coef] * nb_labels, size=nb_clients)  * (nb_points_by_clients+2)).astype(int)# 1 line = 1 worker
    ordered_indices = sorted(unique_values.values(), key=len)
    split = []
    for i in range(nb_clients):
        indices_for_client_i = []
        for j in range(nb_labels):
            indices_by_label = ordered_indices[j]
            indices_for_client_i += random.sample(list(indices_by_label), matrix[i][j]) # Join lists
        split.append(np.array(indices_for_client_i))

    return split


def random_split(Y: np.ndarray, nb_clients: int) -> np.ndarray:
    """Split randomly the dataset."""
    indices = np.arange(len(Y))
    random.shuffle(indices)
    return [indices[i::nb_clients] for i in range(nb_clients)]


def tsne(data: np.ndarray) -> np.ndarray:
    """Compute the TSNE representation of a dataset."""
    np.random.seed(25)
    tsne = TSNE()
    X_embedded = tsne.fit_transform(data)
    return X_embedded


def find_cluster(embedded_data: np.ndarray, nb_cluster: int = 10) -> np.ndarray:
    """Find cluster in a dataset."""
    np.random.seed(25)
    # initialize the weights for each cluster
    weights_init = [1 / nb_cluster] * nb_cluster

    # fit the GMM model and assign the clusters to the data points
    clustering = KMeansConstrained(n_clusters=nb_cluster, size_min=len(embedded_data) // nb_cluster, random_state=0).fit(embedded_data)

    predicted_cluster = clustering.predict(embedded_data)
    split = [np.where(predicted_cluster == i)[0] for i in range(nb_cluster)]

    return split

def tsne_split(X: np.ndarray, nb_clients: int, dataset_name: str) -> np.ndarray:
    """Split the dataset by clustering it TSNE representation (equal-size split)."""

    if not file_exist("pickle/real_dataset/{0}-tsne-split.pkl".format(dataset_name)):

        embedded_data = tsne(X)
        split = find_cluster(embedded_data, nb_clients)
        pickle_saver(split, "pickle/real_dataset/{0}-tsne-split".format(dataset_name))

    else:
        split = pickle_loader("pickle/real_dataset/{0}-tsne-split".format(dataset_name))
    # With the found clusters, splitting data.
    return split


def split_across_clients(dataset: RealLifeDataset, nb_clients: int, heterogeneity: str, dataset_name: str) \
        -> List[RealLifeDataset]:
    """Split the dataset across different clients."""

    if heterogeneity == "dirichlet":
        random_indices = diriclet_split(dataset.Y, nb_clients, dirichlet_coef=0.2)
    if heterogeneity == "tsne":
        random_indices = tsne_split(dataset.X, nb_clients, dataset_name)
    if None:
        random_indices = random_split(dataset.Y, nb_clients)
    datasets = [copy.deepcopy(dataset) for i in range(nb_clients)]
    for i in range(nb_clients):
        datasets[i].keep_sub_set(random_indices[i])
    return datasets
