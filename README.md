# Groundeep Unimodal Training

[![Open Lab 1 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/francesco-cal98/dbn-training/blob/main/cc_lab_01.ipynb)
[![Open Lab 2 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/francesco-cal98/dbn-training/blob/main/cc_lab_02.ipynb)

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
- ✅ Educational notebooks (cc_lab_01, cc_lab_02) with wrapper API for easy teaching

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
├── DBN.py                          # Wrapper for educational notebooks
├── RBM.py                          # Wrapper for educational notebooks
├── cc_lab_01.ipynb                 # Lab 1: DBN training on MNIST
├── cc_lab_02.ipynb                 # Lab 2: Additional experiments
├── test_wrappers.py                # Test suite for wrapper classes
├── tests/                          # Unit tests (TODO)
├── requirements.txt
├── setup.py
├── WRAPPER_INFO.md                 # Documentation for wrapper classes
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

## Educational Notebooks

This repository includes Jupyter notebooks for teaching DBN concepts:

- **[cc_lab_01.ipynb](cc_lab_01.ipynb)** [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/francesco-cal98/dbn-training/blob/main/cc_lab_01.ipynb)
  - Introduction to DBN training on MNIST
  - Iterative training visualization
  - Weight visualization
  - Linear read-out experiments
  - Robustness to noise analysis

- **[cc_lab_02.ipynb](cc_lab_02.ipynb)** [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/francesco-cal98/dbn-training/blob/main/cc_lab_02.ipynb)
  - Advanced experiments

### Using the Labs

**Option 1: Google Colab (Recommended for Students)**

Click the badge above or go to the notebook and click "Open in Colab". The notebook will automatically:
1. Clone this repository
2. Install dependencies
3. Set up the environment

No local installation needed!

**Option 2: Local Installation**

```bash
git clone https://github.com/francesco-cal98/dbn-training.git
cd dbn-training
pip install -r requirements.txt
jupyter notebook cc_lab_01.ipynb
```

### Notebook API

The notebooks use wrapper classes ([DBN.py](DBN.py), [RBM.py](RBM.py)) that provide a simple, clean API:

```python
# In the notebooks - simple API for students
from DBN import DBN

dbn = DBN(visible_units=784, hidden_units=[400, 500, 800], k=1, learning_rate=0.1)
dbn.train_static(train_data, train_labels, num_epochs=50, batch_size=125)
```

For more details on the implementation, see [WRAPPER_INFO.md](WRAPPER_INFO.md) and [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md).

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
