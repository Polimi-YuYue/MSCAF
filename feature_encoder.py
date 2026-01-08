#!/usr/bin/env python
# _*_coding:utf-8 _*_
# @Time: 2026/1/8 14:50
# @Author: Yue Yu
# @School: Politecnico di Milano
# @Email: yyu41474@gmail.com
# @Filmname: feature_encoder.py
# @Software: PyCharm
# @Theme: Fault Diagnosis
import torch
import torch.nn as nn
from torch_geometric.nn import ChebConv, BatchNorm


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=15)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu2 = nn.ReLU(inplace=True)
        self.max2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(128)
        self.relu4 = nn.ReLU(inplace=True)
        self.maxpool = nn.AdaptiveMaxPool1d(4)
        self.fc = nn.Linear(128 * 4, 256)
        self.relu5 = nn.ReLU(inplace=True)
        self.drop = nn.Dropout()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.max2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu4(out)
        out = self.maxpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.relu5(out)
        out = self.drop(out)
        return out


class GGL(torch.nn.Module):
    def __init__(self):
        super(GGL, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(256, 10),
            nn.Sigmoid())

    def forward(self, x):
        x = x.view(x.size(0), -1)
        atrr = self.layer(x)
        values, edge_index = Gen_edge(atrr)
        return values.view(-1), edge_index


def Gen_edge(atrr):
    # Ensure device consistency
    device = atrr.device
    A = torch.mm(atrr, atrr.T)
    maxval, maxind = A.max(axis=1)
    A_norm = A / maxval
    k = A.shape[0]
    values, indices = A_norm.topk(k, dim=1, largest=True, sorted=False)
    edge_index = torch.tensor([[], []], dtype=torch.long, device=device)

    for i in range(indices.shape[0]):
        index_1 = torch.zeros(indices.shape[1], dtype=torch.long, device=device) + i
        index_2 = indices[i]
        sub_index = torch.stack([index_1, index_2])
        edge_index = torch.cat([edge_index, sub_index], axis=1)
    return A_norm, edge_index


class Chev_1(nn.Module):
    def __init__(self, in_channels):
        super(Chev_1, self).__init__()
        self.scale = ChebConv(in_channels, 400, K=1)

    def forward(self, x, edge_index, edge_weight):
        scale = self.scale(x, edge_index, edge_weight)
        return scale


class Chev_2(nn.Module):
    def __init__(self, in_channels):
        super(Chev_2, self).__init__()
        self.scale = ChebConv(in_channels, 100, K=1)

    def forward(self, x, edge_index, edge_weight):
        scale = self.scale(x, edge_index, edge_weight)
        return scale


class GCN(nn.Module):
    def __init__(self, in_channel=256, out_channel=10):
        super(GCN, self).__init__()
        self.atrr = GGL()
        self.conv1 = Chev_1(in_channel)
        self.bn1 = BatchNorm(400)
        self.conv2 = Chev_2(400)
        self.bn2 = BatchNorm(100)
        self.layer5 = nn.Sequential(
            nn.Linear(100, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )

    def forward(self, y):
        edge_atrr, edge_index = self.atrr(y)
        x = self.conv1(y, edge_index, edge_weight=edge_atrr)
        x = self.bn1(x)
        x = self.conv2(x, edge_index, edge_weight=edge_atrr)
        x = self.bn2(x)
        x = x.view(x.size(0), -1)
        x = self.layer5(x)
        return x


class feature_encoder(nn.Module):
    def __init__(self):
        super(feature_encoder, self).__init__()
        self.CNN = CNN()
        self.GCN = GCN()

    def forward(self, x):
        out = self.CNN(x)
        out = self.GCN(out)
        return out


def xavier_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(1.0)


class SingleLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SingleLinear, self).__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU())
        self.clf.apply(xavier_init)

    def forward(self, x):
        return self.clf(x)
