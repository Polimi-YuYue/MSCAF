# MSCAF Fault Diagnosis

Implementation of MSCAF for Cross-domain Fault Diagnosis.

## Requirements
pip install -r requirements.txt

## Data Preparation
Prepare your data in `.npz` format containing:
- `data`: Input sensor data (N, 12, 1024)
- `label`: Labels (N,)

## Usage
Run the training script with your data path:

python train.py --data_path "/path/to/your/data.npz" --epochs 30 --batch_size 32