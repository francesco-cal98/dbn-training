# Google Colab Guide for Students

## Quick Start

### Option 1: Direct Links (Easiest!)

Click these badges to open the notebooks directly in Colab:

- **Lab 1**: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/francesco-cal98/dbn-training/blob/main/cc_lab_01.ipynb)
- **Lab 2**: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/francesco-cal98/dbn-training/blob/main/cc_lab_02.ipynb)

### Option 2: Manual Upload

1. Go to [Google Colab](https://colab.research.google.com)
2. Click **File → Upload notebook**
3. Select `cc_lab_01.ipynb` from this repository
4. The setup will run automatically when you execute the first code cell

---

## How It Works

When you open a notebook in Colab, the **first code cell** will:

```python
🔧 Running on Google Colab - setting up environment...

📥 Cloning repository...
📂 Working directory: /content/groundeep-unimodal-training
📦 Installing dependencies...

✅ Setup complete! Ready to run the notebook.
```

This happens automatically - no action needed from you!

---

## What Gets Installed

The notebooks install only minimal dependencies:
- PyTorch (deep learning framework)
- NumPy (numerical computing)
- Matplotlib (plotting)
- Scikit-learn (machine learning utilities)
- Torchvision (for MNIST dataset)

Everything else is already available in Colab!

---

## Using GPU on Colab

To speed up training (optional but recommended):

1. Click **Runtime → Change runtime type**
2. Select **T4 GPU** under "Hardware accelerator"
3. Click **Save**

The code will automatically detect and use the GPU!

---

## Saving Your Work

**Important**: Colab doesn't save your changes automatically!

### Option 1: Save to Google Drive
1. Click **File → Save a copy in Drive**
2. Your notebook will be saved to your Google Drive
3. Changes are automatically saved there

### Option 2: Download Locally
1. Click **File → Download → Download .ipynb**
2. Save to your computer

---

## Common Issues

### Issue: "Module not found" error
**Solution**: Make sure you ran the **first code cell** (setup cell) before running other cells.

### Issue: Training is slow
**Solution**: Enable GPU (see "Using GPU on Colab" above)

### Issue: Session disconnected
**Solution**: Colab sessions timeout after ~12 hours or 90 minutes of inactivity. Just reconnect and re-run the setup cell.

### Issue: Repository clone failed
**Solution**:
1. Click the folder icon on the left sidebar
2. Delete the `groundeep-unimodal-training` folder if it exists
3. Re-run the setup cell

---

## Understanding the Code

### DBN Construction
```python
dbn = DBN(
    visible_units=784,          # MNIST: 28x28 = 784 pixels
    hidden_units=[400, 500, 800],  # 3 hidden layers
    k=1,                        # Contrastive Divergence steps
    learning_rate=0.1,
    use_gpu=torch.cuda.is_available()  # Auto-detect GPU
)
```

### Training
```python
dbn.train_static(
    mnist_tr.data,      # Training images
    mnist_tr.targets,   # Labels (not used in unsupervised training)
    num_epochs=50,      # Number of training epochs
    batch_size=125      # Mini-batch size
)
```

The training is **iterative**: all layers are updated together at each epoch!

### Accessing Representations
```python
# Get hidden representation from layer 0
hidden_repr, _ = dbn.rbm_layers[0].to_hidden(input_data)

# Access weights for visualization
weights = dbn.rbm_layers[0].W  # Shape: (784, 400)
```

---

## Tips for Success

1. **Run cells in order**: Jupyter notebooks are meant to be run sequentially
2. **Check outputs**: Make sure each cell completes before moving to the next
3. **Read comments**: The notebooks have detailed explanations
4. **Experiment**: Try changing hyperparameters to see what happens!

---

## Need Help?

- Check the main [README.md](README.md) for more details
- See [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md) for technical details
- Ask your instructor if you're stuck!

---

## Local Installation (Alternative)

If you prefer to run locally instead of Colab:

```bash
git clone https://github.com/francesco-cal98/dbn-training.git
cd dbn-training
pip install -r requirements.txt
jupyter notebook cc_lab_01.ipynb
```

Make sure you have Python 3.8+ and PyTorch installed!
