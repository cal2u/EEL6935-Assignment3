import os
import torch
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

def getTrainDev(csv_file):
    df = pd.read_csv(csv_file)
    train_data, dev_data = train_test_split(
        df, test_size=0.1)
    return GenomicsDataset(train_data), GenomicsDataset(dev_data)


class GenomicsDataset(Dataset):
    """Genomics dataset."""

    def __init__(self, df, labels=True):
        """
        Args:
            df (DataFrame): pandas DataFrame with annotations.
        """
        self.data = df
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        mapping = {
            'A': 0,
            'G': 1,
            'C': 2,
            'T': 3
        }
        sequence = self.data.iloc[idx, 1]
        one_hot = torch.zeros(len(sequence)*4)

        for i, ch in enumerate(sequence):
            one_hot[i*4 + mapping[ch]] = 1

        # Return data + label if training
        if self.labels:
            label = self.data.iloc[idx, 2]
            return one_hot, label

        return one_hot
