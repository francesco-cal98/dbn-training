"""
Restricted Boltzmann Machine (RBM)
-----------------------------------
Simple, standalone RBM implementation for educational purposes.
Supports Contrastive Divergence training.

This is a minimal, clean implementation suitable for any dataset.
"""

import torch
import torch.nn as nn
import math


class RBM(nn.Module):
    """
    Restricted Boltzmann Machine with Bernoulli visible and hidden units.

    Implements Contrastive Divergence (CD-k) training algorithm.
    """

    def __init__(
        self,
        visible_units,
        hidden_units,
        learning_rate=0.1,
        momentum=0.5,
        weight_decay=0.0001,
        use_gpu=False
    ):
        """
        Initialize RBM.

        Parameters:
        -----------
        visible_units : int
            Number of visible units (input dimension)
        hidden_units : int
            Number of hidden units
        learning_rate : float
            Learning rate for weight updates
        momentum : float
            Momentum coefficient
        weight_decay : float
            L2 weight decay (regularization)
        use_gpu : bool
            Use CUDA if available
        """
        super().__init__()

        self.visible_units = visible_units
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay

        # Device setup
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

        # Initialize weights and biases
        # W: (visible_units, hidden_units)

        self.W = nn.Parameter(
            torch.nn.init.xavier_normal_(
                torch.empty(visible_units, hidden_units, device=self.device)
            )
    )        
        self.v_bias = nn.Parameter(torch.zeros(visible_units, device=self.device))
        self.h_bias = nn.Parameter(torch.zeros(hidden_units, device=self.device))

        # Momentum buffers
        self.W_momentum = torch.zeros_like(self.W)
        self.v_bias_momentum = torch.zeros_like(self.v_bias)
        self.h_bias_momentum = torch.zeros_like(self.h_bias)

    def sample_hidden(self, v):
        """
        Sample hidden units given visible units.

        Returns:
        --------
        h_prob : torch.Tensor
            Hidden unit probabilities p(h=1|v)
        h_sample : torch.Tensor
            Sampled hidden states
        """
        v = v.to(self.device)
        h_prob = torch.sigmoid(v @ self.W + self.h_bias)
        h_sample = (h_prob > torch.rand_like(h_prob)).float()
        return h_prob, h_sample

    def sample_visible(self, h):
        """
        Sample visible units given hidden units.

        Returns:
        --------
        v_prob : torch.Tensor
            Visible unit probabilities p(v=1|h)
        v_sample : torch.Tensor
            Sampled visible states
        """
        h = h.to(self.device)
        v_prob = torch.sigmoid(h @ self.W.T + self.v_bias)
        v_sample = (v_prob > torch.rand_like(v_prob)).float()
        return v_prob, v_sample

    def to_hidden(self, v):
        """
        Transform visible to hidden (for compatibility with notebooks).

        Returns both probabilities and samples.
        """
        return self.sample_hidden(v)

    def to_visible(self, h):
        """
        Transform hidden to visible (for reconstruction).

        Returns both probabilities and samples.
        """
        return self.sample_visible(h)

    def forward(self, v):
        """Forward pass: visible -> hidden probabilities."""
        h_prob, _ = self.sample_hidden(v)
        return h_prob

    @torch.no_grad()
    def contrastive_divergence(self, v_data, k=1):
        """
        Perform one step of Contrastive Divergence.

        Parameters:
        -----------
        v_data : torch.Tensor
            Visible data (batch_size, visible_units)
        k : int
            Number of Gibbs sampling steps

        Returns:
        --------
        loss : float
            Reconstruction error
        """
        v_data = v_data.to(self.device)
        batch_size = v_data.size(0)

        # Positive phase
        h_prob_pos, h_sample_pos = self.sample_hidden(v_data)

        # Negative phase (k steps of Gibbs sampling)
        v_neg = v_data.clone()
        for _ in range(k):
            h_prob_neg, h_sample_neg = self.sample_hidden(v_neg)
            v_prob_neg, v_neg = self.sample_visible(h_sample_neg)

        # Final hidden for negative phase
        h_prob_neg, _ = self.sample_hidden(v_neg)

        # Compute gradients
        positive_grad = (v_data.T @ h_prob_pos) / batch_size
        negative_grad = (v_neg.T @ h_prob_neg) / batch_size

        dW = positive_grad - negative_grad - self.weight_decay * self.W.data
        dv_bias = (v_data.sum(0) - v_neg.sum(0)) / batch_size
        dh_bias = (h_prob_pos.sum(0) - h_prob_neg.sum(0)) / batch_size

        # Update with momentum (in-place operations)
        self.W_momentum.mul_(self.momentum).add_(dW, alpha=self.learning_rate)
        self.v_bias_momentum.mul_(self.momentum).add_(dv_bias, alpha=self.learning_rate)
        self.h_bias_momentum.mul_(self.momentum).add_(dh_bias, alpha=self.learning_rate)

        self.W.data.add_(self.W_momentum)
        self.v_bias.data.add_(self.v_bias_momentum)
        self.h_bias.data.add_(self.h_bias_momentum)

        # Compute reconstruction error
        loss = float(torch.mean((v_data - v_prob_neg) ** 2).item())

        return loss

    def __repr__(self):
        return (f"RBM({self.visible_units} -> {self.hidden_units}, "
                f"lr={self.learning_rate}, momentum={self.momentum})")
