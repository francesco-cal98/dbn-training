"""
Demo script for PCA Explorer - trains a quick small model for testing.

This script trains a small DBN for just a few epochs to quickly
generate a model file for testing the Streamlit app without waiting
for full training.

WARNING: The representations won't be good (poor class separation),
but you can test the app functionality.
"""

import torch
from torchvision import datasets, transforms
from DBN import DBN

def main():
    print("=" * 60)
    print("DEMO MODE - Quick Training for PCA Explorer Testing")
    print("=" * 60)
    print("\nThis will train a SMALL model for just 5 epochs.")
    print("The model won't have good representations, but you can")
    print("test the Streamlit app functionality.\n")
    print("For production use, run: python train_and_save.py")
    print("=" * 60)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Load MNIST (small subset)
    print("\nLoading MNIST dataset...")
    mnist_tr = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    # Use only 10,000 samples for demo
    subset_size = 10000
    mnist_tr.data = mnist_tr.data[:subset_size]
    mnist_tr.targets = mnist_tr.targets[:subset_size]

    # Initialize SMALL DBN (faster training)
    print("\nInitializing SMALL DBN (demo model)...")
    dbn = DBN(
        visible_units=784,
        hidden_units=[200, 200, 200],  # Smaller layers
        k=1,
        learning_rate=0.1,
        initial_momentum=0.5,
        final_momentum=0.9,
        weight_decay=0.0002,
        use_gpu=torch.cuda.is_available()
    )

    # Quick training (only 5 epochs)
    print("\nTraining for 5 epochs (demo mode)...")
    dbn.train_static(
        train_data=mnist_tr.data,
        train_labels=mnist_tr.targets,
        num_epochs=5,  # Very few epochs
        batch_size=125
    )

    # Save model
    save_path = './dbn_mnist_demo.pth'
    print(f"\nSaving demo model to {save_path}...")
    dbn.save(save_path)

    print("\n" + "=" * 60)
    print("✅ Demo model created!")
    print("=" * 60)
    print("\nTo test the Streamlit app with this demo model:")
    print("1. Edit pca_explorer.py:")
    print("   Change: dbn.load('./dbn_mnist.pth')")
    print("   To:     dbn.load('./dbn_mnist_demo.pth')")
    print("\n2. Run: streamlit run pca_explorer.py")
    print("\n⚠️  Note: Representations won't be good (only 5 epochs).")
    print("For production, train full model: python train_and_save.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
