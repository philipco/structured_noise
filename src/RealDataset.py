import numpy as np
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.datasets import EMNIST, Food101, Places365, Flowers102, EuroSAT

from src.CompressionModel import RandK
from src.PickleHandler import pickle_loader, pickle_saver
from src.SyntheticDataset import AbstractDataset
from src.Utilities import get_project_root, file_exist, create_folder_if_not_existing


def get_path_to_datasets() -> str:
    """Return the path to the datasets. For sake of anonymization, the path to datasets on clusters is not keep on
    GitHub and must be personalized locally"""
    return get_project_root()


class RealLifeDataset(AbstractDataset):

    def __init__(self, dataset_name: str, omega=None):
        super().__init__(dataset_name)
        if file_exist("pickle/real_dataset/{0}.pkl".format(dataset_name)):
            myinstance = pickle_loader("pickle/real_dataset/{0}".format(dataset_name))
            for k in myinstance.__dict__.keys():
                setattr(self, k, getattr(myinstance, k))
            self.define_compressors(omega)
        else:
            print("Preparing the dataset.")
            self.load_data(dataset_name)
            self.define_compressors(omega)
            create_folder_if_not_existing("pickle/real_dataset/")
            pickle_saver(self, "pickle/real_dataset/{0}".format(dataset_name))
            print("Done dataset preparation.")

    def define_compressors(self, omega=None):
        super().define_compressors(omega=omega)
        self.rand1 = RandK(self.sketcher.sub_dim, dim=self.dim, biased=False)
        print("New omega rand1:", self.rand1.omega_c)


    def load_data(self, dataset_name):
        path_to_dataset = '{0}/../../DATASETS/'.format(get_path_to_datasets())
        if dataset_name == 'cifar10':

            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

            transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ])

            train_data = datasets.CIFAR10(root=path_to_dataset, train=True, download=True, transform=transform_train)
            flat_data = np.reshape(train_data.data, (len(train_data), -1))

        if dataset_name == 'cifar100':

            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

            transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ])

            train_data = datasets.CIFAR100(root=path_to_dataset, train=True, download=True, transform=transform_train)
            flat_data = np.reshape(train_data.data, (len(train_data), -1))

        elif dataset_name == 'mnist':
            # Normalization see : https://stackoverflow.com/a/67233938
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            train_data = datasets.MNIST(root=path_to_dataset, train=True, download=False, transform=transform)
            flat_data = np.reshape(train_data.data.numpy(), (len(train_data), -1))

        elif dataset_name == "fashion_mnist":

            train_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            train_data = datasets.FashionMNIST(path_to_dataset, download=True, train=True, transform=train_transforms)
            flat_data = np.reshape(train_data.data.numpy(), (len(train_data), -1))

        elif dataset_name == "EuroSAT":
            transform = transforms.Compose([
                transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,), (0.5,))]
            )
            train_data = EuroSAT("../../DATASETS", download=False, transform=transform)
            data = []
            print("Getting all pictures from the EuroSAT dataset.")
            for i in range(len(train_data)):
                data.append(np.reshape(train_data[i][0].numpy(), -1))
            flat_data = np.array(data)

        elif dataset_name == "emnist":
            transform = transforms.Compose([
                transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
            )
            train_data = EMNIST("./dataset", train=True, download=True, transform=transform, split="balanced")
            flat_data = np.reshape(train_data.data.numpy(), (len(train_data), -1))

        elif dataset_name == "Food101":
            transform = transforms.Compose([
                transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,), (0.5))]
            )
            train_data = Food101("./dataset", download=True, transform=transform, split="train")
            flat_data = np.reshape(train_data.data.numpy(), (len(train_data), -1))

        elif dataset_name == "Places365":
            transform = transforms.Compose([
                transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,), (0.5))]
            )
            train_data = Places365("./dataset", download=True, transform=transform, split="train-standard", small=True)
            flat_data = np.reshape(train_data.data.numpy(), (len(train_data), -1))

        elif dataset_name == "Flowers102":
            img_size = 32
            imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            transform = transforms.Compose([transforms.Resize(img_size),
                                           transforms.Pad(8, padding_mode='reflect'),
                                           transforms.RandomCrop(img_size),
                                           transforms.ToTensor(),
                                           transforms.Normalize(*imagenet_stats)])
            train_data = Flowers102("./dataset", download=True, transform=transform, split="train")
            data = []
            print("Getting all pictures from the Flowers102 dataset.")
            for i in range(len(train_data)):
                data.append(np.reshape(train_data[i][0].numpy(), -1))
            flat_data = np.array(data)
        else:
            raise ValueError("{0} unknown".format(dataset_name))


        self.dim = len(flat_data[0])
        self.size_dataset = len(flat_data)
        standardize_data = StandardScaler().fit_transform(flat_data)
        print("Mean:", np.mean(standardize_data))
        print("Standard deviation:", np.std(standardize_data))
        self.upper_sigma = np.cov(standardize_data.T)
        eig, eigvectors = np.linalg.eig(self.upper_sigma)
        big_eig = (eig.real > 10**-14).sum()
        if big_eig != self.dim:
            print("Warning: there is {0} eigenvalues that are smaller than 10^-14".format(self.dim - big_eig))
            pca = decomposition.PCA(big_eig)
            self.X_complete = pca.fit_transform(standardize_data)
            self.upper_sigma = np.cov(self.X_complete.T)
            self.dim = big_eig
        else:
            self.X_complete = standardize_data

        self.upper_sigma_inv = np.linalg.inv(self.upper_sigma)
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

