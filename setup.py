from setuptools import setup, find_packages

setup(
    name="groundeep-unimodal",
    version="0.1.0",
    description="Unimodal training pipeline for iDBN (independent Deep Belief Networks)",
    author="Your Name",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "wandb>=0.15.0",
        "PyYAML>=6.0",
        "tqdm>=4.65.0",
    ],
    entry_points={
        "console_scripts": [
            "train-idbn=src.main_scripts.train:run_training",
        ],
    },
)
