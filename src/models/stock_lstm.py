import torch
import torch.nn as nn

class StockAttentionLSTM(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=128, num_layers=2, output_dim=3, dropout=0.3):
        super(StockAttentionLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Fix: Positional Encoding for M=7 days
        self.pos_encoder = nn.Parameter(torch.randn(1, 7, hidden_dim))

        self.layernorm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True,
            dropout=0.1
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        # x: [batch, 7, 7]
        lstm_out, _ = self.lstm(x) # [batch, 7, 128]

        # Add temporal awareness
        x = lstm_out + self.pos_encoder

        # Pre-Norm Attention
        norm_x = self.layernorm(x)
        attn_out, _ = self.attention(norm_x, norm_x, norm_x)
        x = x + attn_out

        # Fix: Look at all M=7 days via pooling instead of just x[:, -1, :]
        x = x.mean(dim=1)

        return self.classifier(x)