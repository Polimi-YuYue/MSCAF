#!/usr/bin/env python
# _*_coding:utf-8 _*_
# @Time: 2026/1/8 14:46
# @Author: Yue Yu
# @School: Politecnico di Milano
# @Email: yyu41474@gmail.com
# @Filmname: train.py
# @Software: PyCharm
# @Theme: Fault Diagnosis

import os
import argparse
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import trange

# Custom imports
from data_transform import load_data
from mymodel import MSCAF
from utils import AverageMeter

# Optimization for certain environments (Optional: keep if needed for your machine)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def parse_args():
    parser = argparse.ArgumentParser(description="MSCAF Fault Diagnosis Training")

    # Data arguments
    parser.add_argument('--data_path', type=str, required=True, help='Path to the .npz data file')
    parser.add_argument('--save_dir', type=str, default='./results', help='Directory to save results')

    # Model arguments
    parser.add_argument('--num_sensors', type=int, required=True, help='Number of sensor inputs')
    parser.add_argument('--input_dim', type=int, required=True, help='Input dimension per sensor')
    parser.add_argument('--num_classes', type=int, required=True, help='Number of fault classes')

    # Hyperparameters
    parser.add_argument('--alpha', type=float, required=True, help='Dirichlet distribution parameter')
    parser.add_argument('--beta', type=float, required=True, help='Loss weight parameter')
    parser.add_argument('--epochs', type=int, required=True, help='Number of training epochs')
    parser.add_argument('--lambda_epochs', type=int, required=True, help='Annealing epoch threshold')
    parser.add_argument('--lr', type=float, required=True, help='Learning rate')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size')
    parser.add_argument('--test_split', type=float, required=True, help='Validation split ratio')

    return parser.parse_args()


@torch.no_grad()
def valid(model, test_loader, global_step, views, device):
    model.eval()
    loss_meter = AverageMeter()
    correct_num, data_num = 0, 0

    for batch_idx, (data, target) in enumerate(test_loader):
        # Prepare data structure: [batch, sensors, features] -> list of [batch, 1, features]
        data = torch.squeeze(data, dim=3)
        data = [data[:, i, :].unsqueeze(1).to(device) for i in range(data.shape[1])]
        target = target.to(device)

        data_num += target.size(0)

        evidence, evidence_a, loss = model(data, target, global_step, mode='test')
        _, predicted = torch.max(evidence_a.data, 1)
        correct_num += (predicted == target).sum().item()

        loss_meter.update(loss.item())

    return loss_meter.avg, correct_num / data_num


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Load Data
    print(f"Loading data from {args.data_path}...")
    try:
        data_file = np.load(args.data_path)
        in_ = data_file['data']
        out_ = data_file['label']
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Data Preprocessing
    in_ = in_[:, np.newaxis]  # Add channel dimension if needed
    x_train, x_test, y_train, y_test = train_test_split(in_, out_, test_size=args.test_split, random_state=42)

    train_loader = load_data(x_train, y_train, batch_size=args.batch_size)
    test_loader = load_data(x_test, y_test, batch_size=args.batch_size)

    # Initialize Model
    in_dims = [args.input_dim] * args.num_sensors
    model = MSCAF(in_dims, args.num_classes, args.alpha, args.beta, args.lambda_epochs).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)

    # Training Loop
    print("Start Training...")
    for epoch in trange(args.epochs, desc='Training', unit='epoch'):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # Prepare data
            data = torch.squeeze(data, dim=3)
            data_list = [data[:, i, :].unsqueeze(1).to(device) for i in range(data.shape[1])]
            target = target.to(device)

            # Forward & Backward
            evidence, evidence_a, loss = model(data_list, target, epoch, mode='train')
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Validation
        train_loss, train_acc = valid(model, train_loader, args.lambda_epochs, args.num_sensors, device)
        test_loss, test_acc = valid(model, test_loader, args.lambda_epochs, args.num_sensors, device)

        print(
            f"\nEpoch: {epoch} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {test_loss:.4f} Acc: {test_acc:.4f}")

    # Save evidence tracking (optional)
    save_path = os.path.join(args.save_dir, 'evidence_tracking.xlsx')
    model.save_evidence_to_excel(save_path)


if __name__ == "__main__":
    main()