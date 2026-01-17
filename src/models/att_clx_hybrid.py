import torch
import torch.nn as nn
import torch.nn.functional as F


class AttCLXHybrid(nn.Module):
    def __init__(self, input_dim=7, time_steps=20, lstm_units=64, output_dim=3):
        """
        Implementation of the Attention-based CNN-LSTM model.

        Args:
            input_dim (int): Number of features (OHLCV + Amount + ARIMA Residuals).
            time_steps (int): Size of the sliding window (Look-back period).
            lstm_units (int): Number of hidden units in the LSTM layer.
            output_dim (int): Number of classification classes.
        """
        super(AttCLXHybrid, self).__init__()

        # 1. CNN Layer: filters=64, kernel_size=1, activation='relu'
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=1)
        self.dropout1 = nn.Dropout(0.3)

        # 2. Bidirectional LSTM: return_sequences=True
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=lstm_units,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.dropout2 = nn.Dropout(0.3)

        # 3. Attention Block (attention_3d_block logic)
        # Bi-LSTM output is (batch, time_steps, lstm_units * 2)
        # Dense layer applied to the time dimension for each feature
        self.attention_dense = nn.Linear(time_steps, time_steps)

        # 4. Flatten + Final Output
        # The flattened size is exactly time_steps * (lstm_units * 2)
        self.flatten_size = time_steps * (lstm_units * 2)
        self.classifier = nn.Linear(self.flatten_size, output_dim)

    def forward(self, x):
        # x shape: [batch, time_steps, input_dim]

        # Conv1D expects [batch, channels, length]
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = self.dropout1(x)

        # LSTM expects [batch, length, channels]
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)  # Output shape: [batch, 20, 128]
        lstm_out = self.dropout2(lstm_out)

        # --- Attention Logic ---
        # Permute to apply Dense over the time dimension (axis 1 in Keras)
        # lstm_out: [B, T, D] -> [B, D, T]
        a = lstm_out.transpose(1, 2)
        a = self.attention_dense(a)
        a_probs = torch.softmax(a, dim=-1)  # Softmax over the time dimension

        # Permute back: [B, D, T] -> [B, T, D]
        a_probs = a_probs.transpose(1, 2)

        # Element-wise multiplication (Multiply() in Keras)
        attn_mul = lstm_out * a_probs
        # ------------------------

        # Flatten
        attn_mul = attn_mul.reshape(attn_mul.size(0), -1)

        # Final Dense Layer
        return self.classifier(attn_mul)