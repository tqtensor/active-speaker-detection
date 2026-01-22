import torch.nn as nn
from torch.nn import MultiheadAttention
from torch.nn import functional as F


class attentionLayer(nn.Module):
    """A transformer-based attention layer with feedforward network.

    Implements a cross-attention mechanism followed by a feedforward network
    with layer normalization and residual connections.

    Attributes:
        self_attn: Multi-head attention module for cross-attention.
        linear1: First linear transformation in the feedforward network.
        linear2: Second linear transformation in the feedforward network.
        norm1: Layer normalization after attention.
        norm2: Layer normalization after feedforward network.
        dropout: Dropout layer for feedforward network.
        dropout1: Dropout layer after attention.
        dropout2: Dropout layer after feedforward network.
        activation: Activation function (ReLU).
    """

    def __init__(self, d_model, nhead, dropout=0.1):
        """Initializes the attentionLayer with cross-attention and feedforward components.

        Args:
            d_model: Dimensionality of the model embeddings.
            nhead: Number of attention heads.
            dropout: Dropout probability for regularization.
        """
        super(attentionLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 4, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, src, tar):
        """Applies cross-attention and feedforward transformation.

        Performs cross-attention from tar (query) to src (key/value), followed
        by a feedforward network with residual connections and layer normalization.

        Args:
            src: Source tensor of shape (B, T, C) where B is batch size,
                T is sequence length, and C is feature dimension.
            tar: Target tensor of shape (B, T, C) used as query for attention.

        Returns:
            Transformed tensor of shape (B, T, C) after attention and
            feedforward processing.
        """
        src = src.transpose(0, 1)  # B, T, C -> T, B, C
        tar = tar.transpose(0, 1)  # B, T, C -> T, B, C
        src2 = self.self_attn(tar, src, src, attn_mask=None, key_padding_mask=None)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        src = src.transpose(0, 1)  # T, B, C -> B, T, C
        return src
