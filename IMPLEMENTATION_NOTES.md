# Implementation Notes: Iterative vs Greedy DBN

## Training Strategy

This implementation uses **iterative training**, not greedy layer-wise training.

### Iterative Training (What We Do)

In iterative training, all RBM layers are updated **simultaneously** during each epoch:

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        v = batch_data
        for rbm in rbm_layers:
            # Forward through layer
            h = rbm.forward(v)
            # Update THIS layer with CD-k
            rbm.contrastive_divergence(v, k=k)
            # Use output as input to next layer
            v = h
```

**Key characteristic**: All layers see updated representations from previous layers at each iteration.

### Greedy Layer-wise Training (Traditional DBN)

In greedy training, each layer is trained **separately** to completion before the next:

```python
for layer_idx, rbm in enumerate(rbm_layers):
    # Train ONLY this layer for ALL epochs
    for epoch in range(num_epochs):
        for batch in dataloader:
            # Transform through FROZEN previous layers
            v = transform_through_previous_layers(batch, layer_idx)
            # Update only this layer
            rbm.contrastive_divergence(v, k=k)
    # This layer is now frozen forever
```

**Key characteristic**: Each layer is fully trained before moving to the next.

---

## Why Iterative?

1. **More flexible**: Layers can adapt to each other throughout training
2. **Better representations**: Earlier layers continue to improve based on later layers
3. **Joint optimization**: All layers work together from the start
4. **Research focus**: This repository is focused on iterative DBN research

---

## Compatibility with Notebooks

The notebooks (cc_lab_01, cc_lab_02) work with both approaches because they only use:
- Construction: `DBN(visible_units, hidden_units, ...)`
- Training: `dbn.train_static(data, labels, epochs, batch_size)`
- Access: `dbn.rbm_layers[i]`, `rbm.W`, `rbm.to_hidden()`

The **interface** is the same; only the **internal training** differs.

---

## Generic Dataset Support

This implementation works with **any dataset**:

### MNIST (notebooks)
```python
dbn = DBN(visible_units=784, hidden_units=[400, 500, 800], ...)
dbn.train_static(mnist_data, mnist_labels, epochs=50, batch_size=125)
```

### Custom Dataset (your research)
```python
dbn = DBN(visible_units=10000, hidden_units=[1500, 1500], ...)

# Works with Tensor
dbn.train_static(custom_data_tensor, labels, epochs=100, batch_size=128)

# Or with DataLoader
for epoch in range(epochs):
    for batch in custom_dataloader:
        # ... manual training loop if needed
```

### Any New Dataset
```python
# Just need: (N, D) tensor where D = visible_units
new_data = torch.randn(1000, 784)  # 1000 samples, 784 dimensions
dbn = DBN(visible_units=784, hidden_units=[100, 50], ...)
dbn.train_static(new_data, None, epochs=20, batch_size=64)
```

---

## No Special Requirements

Unlike `src/classes/gdbn_model.py` which expects:
- Specific dataloader with `cumArea_list`, `CH_list`, `density_list`
- W&B run object for logging
- Validation loader for PCA/probes

This implementation (`DBN.py`, `RBM.py`) needs only:
- Input data as `torch.Tensor` or `DataLoader`
- Dimensions and hyperparameters
- Nothing else!

---

## Performance Comparison

| Aspect | Greedy | Iterative |
|--------|--------|-----------|
| Training time | Faster (single pass per layer) | Slower (all layers every iteration) |
| Final quality | Good | Often better (layers co-adapt) |
| Flexibility | Fixed hierarchy | Adaptive hierarchy |
| Research use | Traditional DBN | Modern iDBN |

---

## Usage in This Repository

- **Notebooks** (`cc_lab_01.ipynb`, `cc_lab_02.ipynb`): Use `DBN.py`/`RBM.py` (iterative, generic)
- **Research scripts** (`src/main_scripts/train.py`): Use `src/classes/gdbn_model.py` (iterative, feature-rich)

Both are iterative, but research scripts have additional logging/analysis features.
