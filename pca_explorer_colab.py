"""
PCA Explorer for Google Colab
------------------------------
Modified version of pca_explorer.py that works in Google Colab notebooks.

Usage in Colab:
    Run this cell after training your DBN:

    !streamlit run pca_explorer_colab.py &>/content/logs.txt &
    !npx localtunnel --port 8501

Then click the URL that appears (will be like https://xxx.loca.lt)
"""

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torchvision import datasets, transforms
from DBN import DBN
import plotly.graph_objects as go
import os

# Detect if running on Colab
IN_COLAB = 'COLAB_GPU' in os.environ or 'google.colab' in str(get_ipython())

# Set page config
st.set_page_config(page_title="DBN PCA Explorer", layout="wide")

# Title
st.title("🔍 DBN Representation Explorer")
st.markdown("Interactive visualization of Deep Belief Network learned representations")

if IN_COLAB:
    st.info("🌐 Running on Google Colab")

# Sidebar controls
st.sidebar.header("Controls")

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.sidebar.info(f"Using device: {device}")

# Layer selection
layer_idx = st.sidebar.selectbox(
    "Select Layer",
    options=[0, 1, 2],
    format_func=lambda x: f"Layer {x+1}",
    index=2
)

# Number of samples to visualize
n_samples = st.sidebar.slider(
    "Number of samples",
    min_value=100,
    max_value=5000,  # Reduced for Colab
    value=1000,
    step=100
)

# Cache data loading
@st.cache_data
def load_mnist():
    """Load MNIST dataset."""
    # Use /content for Colab
    data_dir = '/content/data' if IN_COLAB else './data'

    mnist = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )
    return mnist

@st.cache_resource
def load_trained_dbn():
    """Load trained DBN model."""
    # Initialize DBN with same architecture as training
    dbn = DBN(
        visible_units=784,
        hidden_units=[400, 500, 800],
        k=1,
        learning_rate=0.1,
        use_gpu=torch.cuda.is_available()
    )

    # Try to load saved model (check multiple paths and formats)
    model_paths = [
        './dbn_mnist.pkl',
        './dbn_mnist.pth',
        '/content/dbn_mnist.pkl',
        '/content/dbn_mnist.pth',
        '/content/groundeep-unimodal-training/dbn_mnist.pkl',
        '/content/groundeep-unimodal-training/dbn_mnist.pth'
    ]

    loaded = False
    for model_path in model_paths:
        try:
            dbn.load(model_path)
            st.sidebar.success(f"✅ Model loaded from {model_path}")
            loaded = True
            break
        except (FileNotFoundError, Exception):
            continue

    if not loaded:
        st.sidebar.warning("⚠️ No saved model found. Using untrained model.")

    return dbn

def get_kth_layer_repr(input, k, device):
  flattened_input = input.view((input.shape[0], -1)).type(torch.FloatTensor).to(device)
  hidden_repr, __ = dbn_mnist.rbm_layers[k].to_hidden(flattened_input)  # here we access the RBM object
  return hidden_repr

@st.cache_data
def compute_pca_embeddings(_dbn, _mnist_data, _labels, layer_idx, n_samples, device_str):
    """Compute PCA embeddings for given layer."""
    device = torch.device(device_str)

    # Get subset of data
    data_subset = _mnist_data[:n_samples]
    labels_subset = _labels[:n_samples]

    # Get layer representation (CORRECTED)
    hidden_repr = get_kth_layer_repr(_dbn, data_subset, layer_idx, device)

    # Compute PCA
    pca = PCA(n_components=2)
    embeddings = pca.fit_transform(hidden_repr)

    explained_var = pca.explained_variance_ratio_

    return embeddings, labels_subset.numpy(), explained_var, data_subset.numpy()

# Load data
with st.spinner("Loading MNIST..."):
    mnist = load_mnist()

with st.spinner("Loading DBN model..."):
    dbn = load_trained_dbn()

# Compute PCA embeddings
with st.spinner(f"Computing PCA for Layer {layer_idx+1}..."):
    embeddings, labels, explained_var, original_images = compute_pca_embeddings(
        dbn,
        mnist.data,
        mnist.targets,
        layer_idx,
        n_samples,
        str(device)
    )

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"PCA Visualization - Layer {layer_idx+1}")

    # Create interactive plotly scatter plot
    fig = go.Figure()

    # Add scatter plot for each digit class
    for digit in range(10):
        mask = labels == digit
        fig.add_trace(go.Scatter(
            x=embeddings[mask, 0],
            y=embeddings[mask, 1],
            mode='markers',
            name=f'Digit {digit}',
            marker=dict(size=5, opacity=0.6),
            customdata=np.where(mask)[0],
            hovertemplate='<b>Digit %{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>',
            text=[str(digit)] * mask.sum()
        ))

    fig.update_layout(
        xaxis_title=f'PC1 ({explained_var[0]:.1%} variance)',
        yaxis_title=f'PC2 ({explained_var[1]:.1%} variance)',
        height=600,
        hovermode='closest',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    st.plotly_chart(fig, use_container_width=True, key="pca_plot")

with col2:
    st.subheader("Selected Sample")

    # Manual sample selection
    sample_idx = st.number_input(
        "Sample Index",
        min_value=0,
        max_value=n_samples-1,
        value=0,
        step=1
    )

    # Display selected image
    img = original_images[sample_idx].reshape(28, 28)
    label = labels[sample_idx]

    fig_img, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img, cmap='gray')
    ax.set_title(f'Label: {label}', fontsize=16, fontweight='bold')
    ax.axis('off')
    st.pyplot(fig_img)
    plt.close()

    # Display coordinates
    st.markdown("**PCA Coordinates:**")
    st.write(f"PC1: {embeddings[sample_idx, 0]:.3f}")
    st.write(f"PC2: {embeddings[sample_idx, 1]:.3f}")

# Statistics section
st.markdown("---")
st.subheader("📊 Statistics")

col_stat1, col_stat2, col_stat3 = st.columns(3)

with col_stat1:
    st.metric("Total Samples", n_samples)

with col_stat2:
    st.metric("Layer Dimension", dbn.hidden_units[layer_idx])

with col_stat3:
    st.metric("Total Variance Explained", f"{sum(explained_var):.1%}")

# Class distribution
st.subheader("Class Distribution")
unique, counts = np.unique(labels, return_counts=True)
fig_dist = go.Figure(data=[go.Bar(x=unique, y=counts)])
fig_dist.update_layout(
    xaxis_title="Digit",
    yaxis_title="Count",
    height=300
)
st.plotly_chart(fig_dist, use_container_width=True)

# Colab-specific instructions
if IN_COLAB:
    with st.expander("📱 Running on Google Colab"):
        st.markdown("""
        ### How to Use on Colab

        This app is running through a LocalTunnel proxy. If the connection drops:

        1. Stop the current cells
        2. Re-run:
           ```python
           !streamlit run pca_explorer_colab.py &>/content/logs.txt &
           !npx localtunnel --port 8501
           ```
        3. Click the new URL

        ### Limitations on Colab
        - Maximum 5,000 samples (to avoid memory issues)
        - Session will timeout after ~12 hours
        - May need to re-authenticate the tunnel periodically
        """)

# Information panel
with st.expander("ℹ️ How to use this app"):
    st.markdown("""
    ### Navigation
    - **Layer Selection**: Choose which DBN layer to visualize (Layer 1, 2, or 3)
    - **Sample Count**: Adjust the number of MNIST samples to include
    - **Sample Index**: Enter a specific sample number to view

    ### Interaction
    - Hover over points to see digit labels and coordinates
    - Use the legend to toggle visibility of different digit classes
    - Zoom and pan using plotly's built-in controls

    ### Interpretation
    - Well-separated clusters indicate good learned representations
    - Points close together have similar hidden representations
    - Variance explained shows information retained in 2D
    """)

# Technical details
with st.expander("🔧 Technical Details"):
    st.markdown(f"""
    ### Model Architecture
    - **Input**: 784 (28×28 MNIST images)
    - **Layer 1**: {dbn.hidden_units[0]} units
    - **Layer 2**: {dbn.hidden_units[1]} units
    - **Layer 3**: {dbn.hidden_units[2]} units

    ### PCA Settings
    - **Components**: 2
    - **Layer {layer_idx+1} Variance Explained**:
      - PC1: {explained_var[0]:.2%}
      - PC2: {explained_var[1]:.2%}
      - Total: {sum(explained_var):.2%}

    ### Training Details
    - **Algorithm**: Iterative DBN
    - **CD-k**: k={dbn.k}
    - **Learning Rate**: {dbn.learning_rate}
    - **Device**: {device}
    """)
