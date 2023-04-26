import numpy as np
from numpy.random import multivariate_normal
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.datasets import EMNIST, Food101, Places365, Flowers102, EuroSAT

from src.CompressionModel import RandK
from src.PytorchDatasetClass import QuantumDataset, A9ADataset, PhishingDataset
from src.utilities.PickleHandler import pickle_loader, pickle_saver
from src.SyntheticDataset import AbstractDataset
from src.utilities.Utilities import file_exist, create_folder_if_not_existing, get_path_to_datasets


IMG_SIZE = 16
resize = transforms.Resize((IMG_SIZE, IMG_SIZE))


class RealLifeDataset(AbstractDataset):

    def __init__(self, dataset_name: str, s=None):
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
            self.define_constants()
            self.w_star = None
            self.real_dataset = True
            create_folder_if_not_existing("pickle/real_dataset/")
            pickle_saver(self, "pickle/real_dataset/{0}".format(dataset_name))
            print("Done dataset preparation.")

    def define_constants(self, w0_seed: int = 42):
        if w0_seed is not None:
            self.w0 = np.zeros(self.dim)
        else:
            self.w0 = multivariate_normal(np.zeros(self.dim), np.identity(self.dim) /self.dim)

    def define_compressors(self, s=None):
        super().define_compressors(s=s)
        self.rand1 = RandK(self.sketcher.sub_dim, dim=self.dim, biased=False)
        print("New omega rand1:", self.rand1.omega_c)

    def regenerate_dataset(self):
        # Concatenate X_complete and Y along the last axis
        data = np.concatenate((self.X_complete, self.Y.reshape(-1, 1)), axis=-1)

        # Shuffle the data
        np.random.shuffle(data)

        # Split X_complete and Y again
        self.X_complete = data[:, :-1]
        self.Y = data[:, -1].astype(np.int64)


    def load_data(self, dataset_name):
        path_to_dataset = get_path_to_datasets()
        if dataset_name == 'cifar10':

            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

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

        elif dataset_name == "a9a":
            train_data = A9ADataset()

        elif dataset_name == "phishing":
            train_data = PhishingDataset()

        elif dataset_name == "euroSAT":
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                resize,
                transforms.Grayscale(),
            ])
            train_data = EuroSAT("../../DATASETS", download=False, transform=transform_train)

        elif dataset_name == "Food101":
            transform = transforms.Compose([resize, transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,), (0.5))])
            train_data = Food101("./dataset", download=True, transform=transform, split="train")
            # flat_data = np.reshape(train_data.data.numpy(), (len(train_data), -1))

        elif dataset_name == "Places365":
            transform = transforms.Compose([
                resize, transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,), (0.5))]
            )
            train_data = Places365("./dataset", download=True, transform=transform, split="train-standard", small=True)
            # flat_data = np.reshape(train_data.data.numpy(), (len(train_data), -1))

        elif dataset_name == "flowers102":
            imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

            transform_train = transforms.Compose([
                transforms.ToTensor(),
                resize,
                transforms.Grayscale(),
            ])

            train_data = Flowers102("./dataset", download=True, transform=transform_train, split="train")
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

        self.X_without_std = flat_data
        standardize_data = StandardScaler().fit_transform(flat_data)
        self.X_complete = standardize_data
        print("Mean:", np.mean(standardize_data))
        print("Standard deviation:", np.std(standardize_data))

        self.upper_sigma = np.cov(standardize_data.T)
        eig, eigvectors = np.linalg.eig(self.upper_sigma)
        big_eig = (eig.real > 10**-14).sum()
        eig = np.sort(eig)[::-1]
        if self.dim >= 64:
            print("Eigenvalues - biggest: {0}, 32: {1}, 64: {2}, smallest: {3}".format(eig[0], eig[31], eig[63], eig[big_eig-1]))
        else:
            print("Eigenvalues - biggest: {0}, smallest: {1}".format(eig[0], eig[big_eig - 1]))
        print("Warning: there is {0} eigenvalues that are smaller than 10^-14".format(self.dim - big_eig))
        pca = decomposition.PCA(self.dim)
        self.X_pca = pca.fit_transform(standardize_data)
        self.upper_sigma_pca = np.cov(self.X_pca.T)

        print("Effective dimension:", len(self.X_complete[0]))

        self.upper_sigma_inv = np.linalg.inv(self.upper_sigma)
        self.upper_sigma_inv_pca = np.linalg.inv(self.upper_sigma_pca)

        self.L = eig[0] # biggest eigenvalues
        print("L=", self.L)

        self.trace = np.trace(self.upper_sigma)
        print("Trace H:", self.trace)

        print("Trace H^{-1}:", np.trace(self.upper_sigma_inv))
        print("Computed cov.")

    def string_for_hash(self, nb_runs: int, stochastic: bool = False, batch_size: int = 1):
        hash = "{0}runs-N{1}-D{2}".format(nb_runs, self.size_dataset, self.dim)
        if self.name:
            hash = "{0}-{1}".format(self.name, hash)
        if not stochastic:
            hash = "{0}-full".format(hash)
        elif batch_size != 1:
            hash = "{0}-b{1}".format(hash, batch_size)
        return hash

