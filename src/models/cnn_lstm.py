import torch
import torch.nn as nn

INPUT_DIM = 7 # OHLCV, distance in % from 14-days MA, 14-days ATR in %
CLASSES = 3   # Classify to either "buy", "neutral", or "sell" based on ATR

class StockCNNLSTM(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=64, output_dim=CLASSES, dropout=0.2):
        """
        Exact implementation of alexkalinins/cnn-lstm-stock
        Modified for M=60 lookback and PyTorch.
        """
        super(StockCNNLSTM, self).__init__()

        # 1. 1D Convolutional Layer
        # Added padding=2 to keep the sequence length at 60 so it can be added to the input
        self.conv1d = nn.Conv1d(
            in_channels=input_dim,
            out_channels=32,
            kernel_size=5,
            padding=2
        )
        self.swish = nn.SiLU()

        # Skip Connection Projection
        # Projects 5 features to 32 features so we can perform element-wise addition
        self.skip_proj = nn.Linear(input_dim, 32)

        # 2. LSTM Layer
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        # 3. Dense Output Layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: [batch, 60, 7]  <-- Updated comment to reflect 7 features

        # Path A: CNN Feature Extraction
        x_cnn = x.transpose(1, 2)
        x_conv = self.swish(self.conv1d(x_cnn))
        x_conv = x_conv.transpose(1, 2)

        # Path B: Skip Connection
        # This proj layer now correctly handles 7 -> 32
        x_skip = self.skip_proj(x)

        x_lstm_in = x_conv + x_skip

        # LSTM processing
        _, (hn, _) = self.lstm(x_lstm_in)

        # Final Classification
        out = self.dropout(hn[-1])
        return self.fc(out)