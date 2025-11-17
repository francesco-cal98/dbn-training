"""
Deep Belief Network (DBN) with Iterative Training
--------------------------------------------------
Iterative DBN implementation where all layers are updated together
during training (not greedy layer-wise).

This is a clean, minimal implementation suitable for any dataset.
"""

import torch
import pickle
from torch.utils.data import TensorDataset, DataLoader
from RBM import RBM


class DBN:
    """
    Deep Belief Network with iterative training.

    Unlike greedy layer-wise training, this updates all RBM layers
    simultaneously during each training iteration.
    """

    def __init__(
        self,
        visible_units,
        hidden_units,
        k=1,
        learning_rate=0.1,
        learning_rate_decay=False,
        initial_momentum=0.5,
        final_momentum=0.9,
        weight_decay=0.0001,
        xavier_init=False,
        increase_to_cd_k=False,
        use_gpu=False
    ):
        """
        Initialize iterative DBN.

        Parameters:
        -----------
        visible_units : int
            Number of visible units (input dimension)
        hidden_units : list of int
            List of hidden layer sizes, e.g., [400, 500, 800]
        k : int
            Number of Contrastive Divergence steps
        learning_rate : float
            Learning rate for all layers
        learning_rate_decay : bool
            Whether to apply learning rate decay (not implemented)
        initial_momentum : float
            Initial momentum value
        final_momentum : float
            Final momentum value (switched after epoch 5)
        weight_decay : float
            L2 weight regularization
        xavier_init : bool
            Use Xavier initialization (not implemented, uses default)
        increase_to_cd_k : bool
            Progressive CD-k (not implemented)
        use_gpu : bool
            Use CUDA if available
        """
        self.visible_units = visible_units
        self.hidden_units = hidden_units
        self.k = k
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.initial_momentum = initial_momentum
        self.final_momentum = final_momentum
        self.weight_decay = weight_decay
        self.xavier_init = xavier_init
        self.increase_to_cd_k = increase_to_cd_k
        self.use_gpu = use_gpu

        # Device
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

        # Create RBM layers
        # Layer sizes: [visible, hidden[0], hidden[1], ...]
        self.layer_sizes = [visible_units] + hidden_units
        self.rbm_layers = []

        for i in range(len(hidden_units)):
            rbm = RBM(
                visible_units=self.layer_sizes[i],
                hidden_units=self.layer_sizes[i + 1],
                learning_rate=learning_rate,
                momentum=initial_momentum,
                weight_decay=weight_decay,
                use_gpu=use_gpu
            )
            self.rbm_layers.append(rbm)

        self.num_layers = len(self.rbm_layers)

        print(f"Iterative DBN initialized:")
        print(f"  Architecture: {self.layer_sizes}")
        print(f"  Layers: {self.num_layers}")
        print(f"  CD-k: {self.k}")
        print(f"  Device: {self.device}")

    def train_static(self, train_data, train_labels, num_epochs, batch_size):
        """
        Iterative training - all layers updated together.

        This method provides compatibility with the notebook interface
        while performing iterative (not greedy) training.

        Parameters:
        -----------
        train_data : torch.Tensor
            Training data (num_samples, visible_units)
        train_labels : torch.Tensor
            Training labels (not used in unsupervised training, kept for API compatibility)
        num_epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        """
        print(f"\n{'='*60}")
        print(f"Starting ITERATIVE training")
        print(f"Epochs: {num_epochs}, Batch size: {batch_size}")
        print(f"{'='*60}\n")

        # Prepare data
        train_data = train_data.to(self.device).float()

        # Flatten if needed
        if train_data.dim() > 2:
            train_data = train_data.view(train_data.size(0), -1)

        # Create dataloader
        if train_labels is not None:
            dataset = TensorDataset(train_data, train_labels)
        else:
            dataset = TensorDataset(train_data)

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Training loop
        for epoch in range(num_epochs):
            epoch_losses = [[] for _ in range(self.num_layers)]

            # Update momentum after epoch 5
            if epoch == 5:
                for rbm in self.rbm_layers:
                    rbm.momentum = self.final_momentum
                print(f"Momentum updated to {self.final_momentum}")

            for batch in dataloader:
                # Extract data
                batch_data = batch[0] if isinstance(batch, (list, tuple)) else batch
                batch_data = batch_data.to(self.device)

                # Iterative update: propagate through layers and update each
                v = batch_data
                for layer_idx, rbm in enumerate(self.rbm_layers):
                    # Update this layer with CD-k
                    loss = rbm.contrastive_divergence(v, k=self.k)
                    epoch_losses[layer_idx].append(loss)

                    # Forward for next layer (no grad needed)
                    with torch.no_grad():
                        h_prob = rbm.forward(v)
                        v = h_prob

                # Free GPU memory after each batch
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()

            # Print progress
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                avg_losses = [sum(losses) / len(losses) if losses else 0.0
                             for losses in epoch_losses]
                loss_str = ", ".join([f"L{i+1}: {loss:.4f}" for i, loss in enumerate(avg_losses)])
                print(f"Epoch {epoch:3d}/{num_epochs}: {loss_str}")

        print(f"\n{'='*60}")
        print(f"Iterative training complete!")
        print(f"{'='*60}\n")

    def forward(self, input_data):
        """
        Forward pass through all layers.

        Parameters:
        -----------
        input_data : torch.Tensor
            Input data (batch_size, visible_units)

        Returns:
        --------
        h_prob : torch.Tensor
            Final layer hidden probabilities
        h_sample : torch.Tensor
            Final layer hidden samples
        """
        input_data = input_data.to(self.device)

        # Flatten if needed
        if input_data.dim() > 2:
            input_data = input_data.view(input_data.size(0), -1)

        v = input_data

        # Forward through all layers
        with torch.no_grad():
            for rbm in self.rbm_layers:
                h_prob, h_sample = rbm.to_hidden(v)
                v = h_prob

        return h_prob, h_sample

    def reconstruct(self, input_data):
        """
        Reconstruct input by encoding and decoding through all layers.

        Parameters:
        -----------
        input_data : torch.Tensor
            Input data

        Returns:
        --------
        reconstruction : torch.Tensor
            Reconstructed data
        """
        input_data = input_data.to(self.device)

        # Flatten if needed
        if input_data.dim() > 2:
            input_data = input_data.view(input_data.size(0), -1)

        # Encode: forward through all layers
        v = input_data
        with torch.no_grad():
            for rbm in self.rbm_layers:
                h_prob, _ = rbm.to_hidden(v)
                v = h_prob

            # Decode: backward through all layers
            h = v
            for rbm in reversed(self.rbm_layers):
                v_prob, _ = rbm.to_visible(h)
                h = v_prob

        return v_prob

    def save(self, filepath):
        """
        Save DBN to file using pickle.

        Parameters:
        -----------
        filepath : str
            Path where to save the model (use .pkl extension)
        """
        # Prepare state dictionary
        state = {
            'visible_units': self.visible_units,
            'hidden_units': self.hidden_units,
            'k': self.k,
            'learning_rate': self.learning_rate,
            'learning_rate_decay': self.learning_rate_decay,
            'initial_momentum': self.initial_momentum,
            'final_momentum': self.final_momentum,
            'weight_decay': self.weight_decay,
            'xavier_init': self.xavier_init,
            'increase_to_cd_k': self.increase_to_cd_k,
            'use_gpu': self.use_gpu,
            'rbm_states': [
                {
                    'W': rbm.W.cpu().detach(),
                    'v_bias': rbm.v_bias.cpu().detach(),
                    'h_bias': rbm.h_bias.cpu().detach(),
                }
                for rbm in self.rbm_layers
            ]
        }

        # Save with pickle
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

        print(f"DBN saved to {filepath}")

    def load(self, filepath):
        """
        Load DBN from file (supports both pickle and torch formats).

        Parameters:
        -----------
        filepath : str
            Path to the saved model file (.pkl or .pth)
        """
        # Try to load with pickle first, fallback to torch.load for compatibility
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
        except (pickle.UnpicklingError, EOFError):
            # Fallback to torch.load for old .pth files
            state = torch.load(filepath, map_location=self.device)

        # Load RBM states
        for idx, rbm_state in enumerate(state['rbm_states']):
            self.rbm_layers[idx].W.data = rbm_state['W'].to(self.device)
            self.rbm_layers[idx].v_bias.data = rbm_state['v_bias'].to(self.device)
            self.rbm_layers[idx].h_bias.data = rbm_state['h_bias'].to(self.device)

        print(f"DBN loaded from {filepath}")

    def __repr__(self):
        return (f"DBN(architecture={self.layer_sizes}, "
                f"k={self.k}, lr={self.learning_rate}, device={self.device})")
