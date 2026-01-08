#!/usr/bin/env python
# _*_coding:utf-8 _*_
# @Time: 2026/1/8 14:49
# @Author: Yue Yu
# @School: Politecnico di Milano
# @Email: yyu41474@gmail.com
# @Filmname: mynmodel.py
# @Software: PyCharm
# @Theme: Fault Diagnosis
import pandas as pd
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F

# Adjusted imports assuming flat directory structure
from utils import KL
from feature_encoder import feature_encoder, SingleLinear


class MSCAF(nn.Module):
    def __init__(self, in_dim, num_classes, alpha, beta, lambda_epoch=50, exp_eta=0.5):
        super(MSCAF, self).__init__()
        self.sensors = len(in_dim)
        self.classes = num_classes
        self.alpha = alpha
        self.lambda_epoch = lambda_epoch
        self.exp_eta = exp_eta
        self.beta = beta
        self.in_dim = in_dim
        self.FeatureInforEncoder = feature_encoder()
        self.ModalityClassifier = nn.ModuleList(
            [SingleLinear(256, self.classes) for _ in range(self.sensors)])
        self.softplus = nn.Softplus()

        # Evidence tracker
        self.evidence_tracker = []
        self.iteration_counter = 0

    def ce_loss(self, p, alpha, c, global_step, annealing_step, c_KL=True):
        S = torch.sum(alpha, dim=1, keepdim=True)
        E = alpha - 1
        label = F.one_hot(p, num_classes=c)
        A = torch.mean(torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True))

        if c_KL:
            annealing_coef = min(1, global_step / annealing_step)
            # alp is adjusted alpha based on true label
            alp = E * (1 - label) + 1
            B = annealing_coef * KL(alp, c)
            return A + torch.mean(B)
        return A

    def jousselme_distance(self, m1, m2):
        dis = m1 - m2
        dis = dis.unsqueeze(1)
        return torch.sqrt((torch.bmm(dis, torch.transpose(dis, -1, 1))) / 2)

    def count_DM(self, views):
        dist_matrix = torch.zeros((len(views), len(views), views[0].shape[0]), device=views[0].device)
        for i in range(len(views)):
            for j in range(len(views)):
                dist_matrix[i][j] = self.jousselme_distance(views[i], views[j]).squeeze()
        return dist_matrix.permute(2, 0, 1)

    def forward(self, X, y, global_step, mode='train'):
        assert mode in ['train', 'valid', 'test']
        evidence, project = self.infer(X)
        loss = 0

        alpha = dict()
        step_rate = global_step / self.lambda_epoch

        # Individual view loss
        alpha = dict()
        evidence_u = dict()
        p = dict()

        for sensor in range(self.sensors):
            alpha[sensor] = evidence[sensor] + 1
            loss += self.beta * self.ce_loss(y, alpha[sensor], self.classes, global_step,
                                             self.lambda_epoch,
                                             c_KL=True)

            S = torch.sum(alpha[sensor], dim=1, keepdim=True)
            p[sensor] = alpha[sensor] / S.expand(alpha[sensor].shape)
            evidence_u[sensor] = self.classes / S

        # Combine sensors
        evidence_a, evidence_u_w = self.combine_sensors(p, evidence_u, evidence, global_step, y)
        alpha_a = evidence_a + 1

        loss += min(step_rate, 1) * self.beta * self.ce_loss(y, alpha_a, self.classes, global_step, self.lambda_epoch,
                                                             c_KL=True)

        self.iteration_counter += 1
        return evidence, evidence_a, loss

    def infer(self, input):
        FeatureInfo, ProjectVector, Evidence = dict(), dict(), dict()
        for sensor in range(self.sensors):
            FeatureInfo[sensor] = self.FeatureInforEncoder(input[sensor])
            Evidence[sensor] = self.softplus(self.ModalityClassifier[sensor](FeatureInfo[sensor]))
        return Evidence, ProjectVector

    def combine_sensors(self, alpha, evidence_u, evidence, global_step=None, y=None):
        # Step 1: Distance Matrix
        dist_matrix = self.count_DM(alpha)
        # Step 2: Average Distance
        dist_matrix_avg = torch.sum(dist_matrix, dim=1) / (len(alpha) - 1)
        # Step 3: Global Evidence Distance
        dist_matrix_global_avg = torch.mean(dist_matrix_avg, dim=1)

        # Step 4 & 5: Normalized Uncertainty
        evidence_u_sum = sum(evidence_u.values())
        evidence_u_matrix = torch.stack([evidence_u[i] / evidence_u_sum for i in range(len(evidence_u))],
                                        dim=1).squeeze()

        wight = evidence_u_matrix
        wight_max, _ = torch.max(wight, dim=1)

        # Step 6: Normalized Weights
        mask = dist_matrix_avg < dist_matrix_global_avg[:, None]
        wight_normalize = torch.exp(-((wight_max + 1)[:, None] - wight)) * (~mask) + torch.exp(-wight) * mask
        wight_normalize = wight_normalize / torch.sum(wight_normalize, dim=1)[:, None]

        # Step 7: Fuse Evidence
        evidence_w = torch.zeros((len(alpha), alpha[0].shape[0], evidence[0].shape[1]), device=alpha[0].device)
        evidence_u_w = torch.zeros((len(alpha), evidence_u[0].shape[0]), device=alpha[0].device)

        wight_normalize_exp = torch.ones((alpha[0].shape[0], len(alpha)), device=alpha[0].device)

        for view in range(len(evidence)):
            evidence_w[view] = (evidence[view] * wight_normalize_exp[:, view][:, None])
            evidence_u_w[view] = (evidence_u[view].squeeze() * wight_normalize_exp[:, view])

        if mode := 'train':  # Only record in training if needed, or keep check
            self._record_evidence_u_w(evidence_u_w, global_step, y)

        evidence_w_sum = torch.sum(evidence_w, dim=0)
        e = evidence_w_sum / torch.sum(wight_normalize_exp, dim=1)[:, None]
        return e, torch.sum(evidence_u_w, dim=0)[:, None] / torch.sum(wight_normalize_exp, dim=1)[:, None]

    def _record_evidence_u_w(self, evidence_u_w, global_step=None, y=None):
        evidence_u_w_np = evidence_u_w.detach().cpu().numpy()
        batch_size = evidence_u_w_np.shape[1]
        for batch_idx in range(batch_size):
            record = {
                'iteration': self.iteration_counter,
                'global_step': global_step if global_step is not None else -1,
                'batch_idx': batch_idx,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            for view_idx in range(evidence_u_w_np.shape[0]):
                record[f'sensor_{view_idx + 1}'] = evidence_u_w_np[view_idx, batch_idx]
            if y is not None:
                record['label'] = y[batch_idx].item() if hasattr(y[batch_idx], 'item') else y[batch_idx]
            self.evidence_tracker.append(record)

    def save_evidence_to_excel(self, filepath='evidence_tracking.xlsx'):
        if not self.evidence_tracker:
            return
        try:
            df = pd.DataFrame(self.evidence_tracker)
            df.to_excel(filepath, index=False)
            print(f"Evidence data saved to {filepath}")
        except Exception as e:
            print(f"Error saving Excel: {e}")
