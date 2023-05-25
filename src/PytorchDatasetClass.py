import numpy as np
import pandas as pd
import torch
from sklearn.datasets import load_svmlight_file
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from src.utilities.Utilities import get_project_root, create_folder_if_not_existing, get_path_to_datasets


class QuantumDataset(Dataset):
    """ Create a dataset class for quantum."""

    def __init__(self, train: bool = True):
        X_train, Y_train, dim = prepare_quantum()
        self.train = train
        print('Total number of point:', len(X_train))
        self.data = X_train
        self.labels = Y_train

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.data[index].float(), self.labels[index].float()  # type(torch.LongTensor)


def prepare_quantum() -> [np.ndarray, np.ndarray, int]:
    """Pre-processing of the quantum dataset."""

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
