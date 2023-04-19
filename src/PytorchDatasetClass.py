import random

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import load_svmlight_file
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from src.utilities.Utilities import get_project_root, create_folder_if_not_existing, get_path_to_datasets


class QuantumDataset(Dataset):
    """ Create a dataset class for quantum."""

    def __init__(self, train: bool = True, iid: str = "iid"):
        root = get_project_root()
        bool_iid = True if iid == "iid" else False
        create_folder_if_not_existing("{0}/pickle/quantum-{1}-N20".format(root, iid))
        X_train, Y_train, dim = prepare_quantum(20, data_path="{0}/pickle/".format(root),
                                                         pickle_path="{0}/pickle/quantum-{1}-N20".format(root, iid),
                                                         iid=bool_iid)
        self.train = train
        print('Total number of point:', len(X_train))
        self.data = X_train
        self.labels = Y_train

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.data[index].float(), self.labels[index].float()  # type(torch.LongTensor)


class PhishingDataset(Dataset):
    """ Create a dataset class for phishing."""


    def __init__(self, train=True, iid: str = "iid"):
        root = get_project_root()
        bool_iid = True if iid == "iid" else False
        create_folder_if_not_existing("{0}/pickle/phishing-{1}-N20".format(root, iid))
        X_train, Y_train, dim = prepare_phishing(20, data_path="{0}/pickle/".format(root),
                                                          pickle_path="{0}/pickle/phishing-{1}-N20".format(root, iid),
                                                          iid=bool_iid)




        self.train = train
        print('Total number of point:', len(X_train))
        self.data = X_train
        self.targets = Y_train

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index: int):
        return self.data[index].float(), self.targets[index].float()


class A9ADataset(Dataset):
    """ Create a dataset class for a9a."""

    # Warning: there is 16 eigenvalues that are smaller than 10^-14

    def __init__(self, train=True, iid: str = "iid"):
        root = get_project_root()
        bool_iid = True if iid == "iid" else False

        create_folder_if_not_existing("{0}/pickle/a9a-{1}-N20".format(root, iid))
        X_train, Y_train, dim = prepare_a9a(20, data_path="{0}/pickle/".format(root),
                                                     pickle_path="{0}/pickle/a9a-{1}-N20".format(root, iid),
                                                     iid=bool_iid, test=False)


        self.train = train
        print('Total number of point:', len(X_train))
        self.data = X_train
        self.targets = Y_train

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index: int):
        return self.data[index].float(), self.targets[index].float()  # type(torch.LongTensor)




def prepare_phishing(nb_devices: int, data_path: str, pickle_path: str, iid: bool = True, dirichlet: int = None,
                     double_check: bool =False):

    raw_X, raw_Y = load_svmlight_file("{0}/phishing/phishing.txt".format(get_path_to_datasets()))

    for i in range(len(raw_Y)):
        if raw_Y[i] == 0:
            raw_Y[i] = -1

    raw_data = pd.DataFrame(data=raw_X.todense())
    raw_data["target"] = raw_Y
    dim = len(raw_data.columns)

    X_data = raw_data.loc[:, raw_data.columns != "target"]
    Y_data = raw_data.loc[:, raw_data.columns == "target"]
    X_data = StandardScaler().fit_transform(X_data)

    # Warning: there is 30 eigenvalues that are smaller than 10^-14
    clf = ExtraTreesClassifier(n_estimators=50)
    clf = clf.fit(X_data, Y_data)
    print(np.sort(clf.feature_importances_))
    # To only select based on max_features, set threshold=-np.inf.
    X_new = SelectFromModel(clf, prefit=True, max_features=32).transform(X_data)



    # X_new = SelectKBest(f_classif, k=dim - 30).fit_transform(X_data, Y_data)

    X_tensor = torch.tensor(X_new, dtype=torch.float64)
    Y_tensor = torch.tensor(Y_data.values, dtype=torch.float64)
    return X_tensor, Y_tensor, dim



def prepare_a9a(nb_devices: int, data_path: str, pickle_path: str, iid: bool = True, dirichlet: int = None,
                double_check: bool =False, test: bool = False):

    if not test:
        raw_X, raw_Y = load_svmlight_file("{0}/a9a/a9a.txt".format(get_path_to_datasets()))
        raw_X = raw_X.todense()
    else:
        raw_X, raw_Y = load_svmlight_file("{0}/a9a/a9a_test.txt".format(get_path_to_datasets()))
        raw_X = raw_X.todense()
        raw_X = np.c_[raw_X, np.zeros((len(raw_Y)))]


    for i in range(len(raw_Y)):
        if raw_Y[i] == 0:
            raw_Y[i] = -1

    raw_data = pd.DataFrame(data=raw_X)
    raw_data["target"] = raw_Y
    dim = len(raw_data.columns) - 1

    X_data = raw_data.loc[:, raw_data.columns != "target"]
    Y_data = raw_data.loc[:, raw_data.columns == "target"]
    X_data = StandardScaler().fit_transform(X_data)

    # Warning: there is 16 eigenvalues that are smaller than 10^-14
    clf = ExtraTreesClassifier(n_estimators=50)
    clf = clf.fit(X_data, Y_data)
    print(np.sort(clf.feature_importances_))
    # To only select based on max_features, set threshold=-np.inf.
    X_new = SelectFromModel(clf, prefit=True, max_features=32).transform(X_data)


    X_tensor = torch.tensor(X_new, dtype=torch.float64)
    Y_tensor = torch.tensor(Y_data.values, dtype=torch.float64)

    return X_tensor, Y_tensor, dim + 1 # Because we added one column for the bias


def prepare_quantum(nb_devices: int, data_path: str, pickle_path: str, iid: bool = True, dirichlet: int = None,
                    double_check: bool =False):

    raw_data= pd.read_csv('{0}/quantum/phy_train.csv'.format(get_path_to_datasets()), sep="\t", header=None)

    # Looking for missing values.
    columns_with_missing_values = []
    for col in range(1, len(raw_data.columns)):
        if (not raw_data[raw_data[col] == 999].empty) or (not raw_data[raw_data[col] == 9999].empty):
            columns_with_missing_values.append(col)
    raw_data.drop(raw_data.columns[columns_with_missing_values], axis=1, inplace=True)
    raw_data = raw_data.rename(columns={0: "ID", 1: "state", 80: "nothing"})
    raw_data = raw_data.drop(['ID', 'nothing'], axis=1)
    raw_data.head()

    # Looking for empty columns (with null std).
    small_std = []
    std_data = raw_data.std()
    for i in range(len(raw_data.columns)):
        if std_data.iloc[i] < 1e-5:
            small_std.append(i)
    raw_data.iloc[:, small_std].describe()

    # Removing columns with null std
    raw_data = raw_data.loc[:, (raw_data.std() > 1e-6)]
    dim = len(raw_data.columns) - 1 # The dataset still contains the label

    raw_data = raw_data.replace({'state': {0: -1}})


    X_data = raw_data.loc[:, raw_data.columns != "state"]
    Y_data = raw_data.loc[:, raw_data.columns == "state"]  # We do not scale labels (+/-1).

    # Transforming into torch.FloatTensor
    X_tensor = torch.tensor(X_data.to_numpy(), dtype=torch.float64)
    Y_tensor = torch.tensor(Y_data.values, dtype=torch.float64)

    return X_tensor, Y_tensor, dim + 1 # Because we added one column for the bias
