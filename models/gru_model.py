"""
Base GRU encoder for stock prediction.

This module contains the shared GRU architecture used by both
regression and classification models.
"""

import torch
import torch.nn as nn


class GRUEncoder(nn.Module):
    """
    GRU-based encoder for time series data.
    
    Architecture:
        Input [batch, seq_len, input_size]
          ↓
        GRU layers
          ↓
        Take last hidden state [batch, hidden_size]
          ↓
        Fully connected layer
          ↓
        Output [batch, fc_hidden_size]
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout_gru: float = 0.0,
        dropout_fc: float = 0.0,
        fc_hidden_size: int = 64
    ):
        """
        Initialize GRU encoder.
        
        Args:
            input_size: Number of input features (e.g., 5 for OHLCV)
            hidden_size: Size of GRU hidden state
            num_layers: Number of stacked GRU layers
            dropout_gru: Dropout probability between GRU layers (ignored if num_layers=1)
            dropout_fc: Dropout probability in fully connected layer
            fc_hidden_size: Size of fully connected hidden layer
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_gru if num_layers > 1 else 0.0,
            batch_first=True  # Input shape: [batch, seq, features]
        )
        
        # Fully connected layer after GRU
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, fc_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_fc)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch, seq_len, input_size]
            
        Returns:
            Encoded features of shape [batch, fc_hidden_size]
        """
        # GRU forward pass
        # output: [batch, seq_len, hidden_size]
        # h_n: [num_layers, batch, hidden_size]
        output, h_n = self.gru(x)
        
        # Take the last hidden state from the last layer
        # h_n[-1]: [batch, hidden_size]
        last_hidden = h_n[-1]
        
        # Pass through FC layer
        # [batch, fc_hidden_size]
        features = self.fc(last_hidden)
        
        return features
    
    def get_num_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick test
    print("Testing GRUEncoder...")
    
    # Create model
    model = GRUEncoder(
        input_size=5,
        hidden_size=128,
        num_layers=2,
        dropout_gru=0.2,
        dropout_fc=0.3,
        fc_hidden_size=64
    )
    
    print(f"Model parameters: {model.get_num_parameters():,}")
    
    # Test forward pass
    batch_size = 16
    seq_len = 10
    x = torch.randn(batch_size, seq_len, 5)
    
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    assert output.shape == (batch_size, 64), "Output shape mismatch"
    print("✓ Test passed!")

