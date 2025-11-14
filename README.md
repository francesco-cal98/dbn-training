# Groundeep Unimodal Training

Minimal and clean repository for training **iDBN (independent Deep Belief Networks)** on unimodal visual data.

## Features

- ✅ Clean, standalone implementation of RBM and iDBN
- ✅ Contrastive Divergence (CD-k) training
- ✅ Automatic W&B logging with:
  - Loss tracking
  - Reconstruction visualizations
  - PCA embeddings (2D/3D)
  - Linear probe analysis
- ✅ Support for uniform and Zipfian-distributed datasets
- ✅ Configurable via YAML

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd groundeep-unimodal-training

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

## Quick Start

### 1. Prepare your dataset

Your dataset should be a `.npz` file with the following keys:
- `D`: Image data (N x H x W or N x C x H x W)
- `N_list`: Labels for each sample
- `cumArea_list`: Cumulative area feature
- `CH_list`: Convex hull feature
- `density_list`: (optional) Density feature

### 2. Configure training

Edit [src/configs/training_config.yaml](src/configs/training_config.yaml):

```yaml
# Algorithm: 'i' for iDBN
algorithm: 'i'

# Dataset
dataset_path: '/path/to/your/dataset'
dataset_name: 'your_dataset.npz'
dataset_distribution: 'uniform'  # or 'zipfian'
multimodal_flag: false

# Architecture
layer_sizes_list:
  - [10000, 1500, 1500]  # Input size, hidden layers

# Hyperparameters
learning_rate: 0.1
weight_penalty: 0.0002
init_momentum: 0.5
final_momentum: 0.95
cd_k: 1
epochs: 100
batch_size: 128

# Sparsity (optional)
sparsity: false
sparsity_factor: 0.1

# Saving
save_path: './networks/idbn'
save_name: 'model'
```

### 3. Train the model

```bash
python src/main_scripts/train.py --config src/configs/training_config.yaml
```

Or if installed with setup.py:
```bash
train-idbn --config src/configs/training_config.yaml
```

## Project Structure

```
groundeep-unimodal-training/
├── src/
│   ├── classes/
│   │   └── gdbn_model.py          # RBM and iDBN implementation
│   ├── datasets/
│   │   └── uniform_dataset.py     # Dataset loaders
│   ├── utils/
│   │   ├── wandb_utils.py         # W&B logging (PCA, reconstructions)
│   │   └── probe_utils.py         # Linear probe analysis
│   ├── configs/
│   │   └── training_config.yaml   # Main configuration
│   └── main_scripts/
│       └── train.py               # Training entrypoint
├── tests/                          # Unit tests (TODO)
├── requirements.txt
├── setup.py
└── README.md
```

## Key Components

### RBM (Restricted Boltzmann Machine)
- Bernoulli visible and hidden units
- CD-k and PCD training
- Optional sparsity regularization
- Dynamic learning rate scheduling

### iDBN (independent Deep Belief Network)
- Layer-wise greedy training
- Automatic W&B logging:
  - Loss curves
  - Auto-reconstruction grids
  - PCA visualizations (2D/3D with feature correlations)
  - Linear probe accuracy
- Flexible architecture configuration

## Logging

The training automatically logs to [Weights & Biases](https://wandb.ai):
- **Loss**: Training reconstruction error per epoch
- **Auto-reconstruction**: Visual comparison of original vs reconstructed images
- **PCA embeddings**: 2D/3D visualizations with feature correlations
- **Linear probe**: Separability analysis per layer

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `algorithm` | 'i' for iDBN | 'i' |
| `learning_rate` | Initial learning rate | 0.1 |
| `weight_penalty` | L2 regularization | 0.0002 |
| `cd_k` | Contrastive Divergence steps | 1 |
| `epochs` | Training epochs | 100 |
| `batch_size` | Batch size | 128 |
| `sparsity` | Enable sparsity regularization | false |
| `dataset_distribution` | 'uniform' or 'zipfian' | 'uniform' |

## Workflow: Develop locally, push to separate repo

This repository is designed to be developed in parallel with the main Groundeep repository:

```bash
# Work on the main repo (with branch unimodal-fixing-experiments)
cd /path/to/Groundeep
git checkout unimodal-fixing-experiments

# Make changes locally
# ... edit files ...

# Copy changes to this standalone repo
cp src/classes/gdbn_model.py /path/to/groundeep-unimodal-training/src/classes/
cp src/main_scripts/train.py /path/to/groundeep-unimodal-training/src/main_scripts/

# Commit and push here
cd /path/to/groundeep-unimodal-training
git add .
git commit -m "Update training pipeline"
git push
```

## License

[Add your license here]

## Citation

If you use this code, please cite:

```
[Add citation info]
```
