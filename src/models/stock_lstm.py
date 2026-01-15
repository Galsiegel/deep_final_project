import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os

# PyTorch Core (From Tutorials 04 & 05)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Data Handling (From Tutorials 05 & 07)
from torch.utils.data import DataLoader, Dataset, TensorDataset

# Preprocessing (From Tutorial 08)
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Course-specific configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# --- MODEL DEFINITION (Modular Approach - Tutorial 05 & 07) ---

class StockAttentionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=3, dropout=0.3):
        super(StockAttentionLSTM, self).__init__()

        # 1. Sequential Engine (Tutorial 07)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)

        # 2. Multi-Head Attention (Transformer-style)
        # embed_dim must match LSTM hidden_dim
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)

        # 3. Normalization and Residual (Tutorial 05/08 Good Practice)
        self.layernorm = nn.LayerNorm(hidden_dim)

        # 4. Classification Head
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch, seq_len, features]
        lstm_out, _ = self.lstm(x)

        # Self-Attention: Query=lstm_out, Key=lstm_out, Value=lstm_out
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Residual Connection (Tutorial 05 Skip Connection) + LayerNorm
        # This prevents vanishing gradients in deeper stacks
        x = self.layernorm(lstm_out + attn_out)

        # Take the last time step for classification (Many-to-One)
        last_step = x[:, -1, :]
        out = self.dropout(last_step)
        return self.fc(out)