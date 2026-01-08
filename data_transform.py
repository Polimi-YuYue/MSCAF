#!/usr/bin/env python
# _*_coding:utf-8 _*_
# @Time: 2026/1/8 14:50
# @Author: Yue Yu
# @School: Politecnico di Milano
# @Email: yyu41474@gmail.com
# @Filmname: data_transform.py
# @Software: PyCharm
# @Theme: Fault Diagnosis
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch


class FaultDiagnosisDataset(Dataset):
    def __init__(self, data, label):
        self.len = data.shape[0]
        data = data.astype(np.float32)
        # Reshape: (Batch, Sensors, Features, 1)
        self.x_data = torch.from_numpy(data).view(-1, 12, 1024, 1)
        self.y_data = torch.from_numpy(label).view(-1).long()

    def __getitem__(self, item):
        return self.x_data[item], self.y_data[item]

    def __len__(self):
        return self.len


def load_data(data, label, batch_size=32, drop_last=False):
    dataset = FaultDiagnosisDataset(data, label)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size,
                             shuffle=True, num_workers=0, drop_last=drop_last)
    return data_loader
