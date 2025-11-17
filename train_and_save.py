"""
Quick script to train DBN on MNIST and save for PCA Explorer.

This script trains the DBN with the same settings as the notebook
and saves it for use with the Streamlit PCA explorer app.
"""

import torch
from torchvision import datasets, transforms
from DBN import DBN

def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load MNIST
    print("\nLoading MNIST dataset...")
    mnist_tr = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    # Initialize DBN (same architecture as notebook)
    print("\nInitializing DBN...")
    dbn = DBN(
        visible_units=784,
        hidden_units=[400, 500, 800],
        k=1,
        learning_rate=0.1,
        initial_momentum=0.5,
        final_momentum=0.9,
        weight_decay=0.0002,
        use_gpu=torch.cuda.is_available()
    )

    # Train
    print("\nStarting training...")
    dbn.train_static(
        train_data=mnist_tr.data,
        train_labels=mnist_tr.targets,
        num_epochs=50,
        batch_size=125
    )

    # Save model
    save_path = './dbn_mnist.pth'
    print(f"\nSaving model to {save_path}...")
    dbn.save(save_path)

    print("\n✅ Training complete! You can now run the PCA explorer:")
    print("   streamlit run pca_explorer.py")

if __name__ == "__main__":
    main()
