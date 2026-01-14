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

class StockLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=1, dropout=0.2):
        """
        Stacked LSTM for Binary Classification (Up/Down)
        input_dim: Number of features (e.g., OHLCV = 5)
        hidden_dim: Number of neurons in LSTM hidden layer
        num_layers: Number of stacked LSTM layers
        """
        super(StockLSTM, self).__init__()

        # 1. The LSTM Engine (From Tutorial 07)
        # batch_first=True means input shape is (batch, seq_len, features)
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 2. Regularization (From Tutorial 08)
        self.dropout = nn.Dropout(dropout)

        # 3. Fully Connected Classification Head (From Tutorial 05)
        self.fc = nn.Linear(hidden_dim, output_dim)

        # Sigmoid maps output to [0, 1] for binary classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: [batch, seq_len, features]
        # Tutorial 07: LSTM returns (output, (h_n, c_n))
        lstm_out, (h_n, c_n) = self.lstm(x)

        # We take the output of the VERY LAST time step (the "conclusion" of the sequence)
        last_time_step_out = lstm_out[:, -1, :]

        # Pass through classification head
        out = self.dropout(last_time_step_out)
        out = self.fc(out)
        return self.sigmoid(out)