"""
Regression model for stock price prediction.

Predicts next-day closing price.
"""

import torch
import torch.nn as nn
from models.gru_model import GRUEncoder


class StockRegressionModel(nn.Module):
    """
    GRU-based model for stock price regression.
    
    Predicts the next day's closing price (normalized).
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout_gru: float = 0.2,
        dropout_fc: float = 0.3,
        fc_hidden_size: int = 64
    ):
        """
        Initialize regression model.
        
        Args:
            input_size: Number of input features (5 for OHLCV)
            hidden_size: GRU hidden size
            num_layers: Number of GRU layers
            dropout_gru: Dropout between GRU layers
            dropout_fc: Dropout in FC layers
            fc_hidden_size: Size of FC layer before output
        """
        super().__init__()
        
        # Shared GRU encoder
        self.encoder = GRUEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout_gru=dropout_gru,
            dropout_fc=dropout_fc,
            fc_hidden_size=fc_hidden_size
        )
        
        # Regression head: predict single value
        # Input: GRU features + opening price (fc_hidden_size + 1)
        self.head = nn.Linear(fc_hidden_size + 1, 1)
    
    def forward(self, x: torch.Tensor, day_open: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, seq_len, input_size] - historical OHLCV
            day_open: Opening price tensor [batch, 1] - day i's opening
            
        Returns:
            Predicted prices [batch, 1]
        """
        # Encode sequence
        features = self.encoder(x)  # [batch, fc_hidden_size]
        
        # Concatenate GRU features with opening price
        combined = torch.cat([features, day_open], dim=1)  # [batch, fc_hidden_size + 1]
        
        # Predict price
        output = self.head(combined)
        
        return output
    
    def get_num_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick test
    print("Testing StockRegressionModel...")
    
    model = StockRegressionModel(
        input_size=5,
        hidden_size=128,
        num_layers=2,
        dropout_gru=0.2,
        dropout_fc=0.3,
        fc_hidden_size=64
    )
    
    print(f"Total parameters: {model.get_num_parameters():,}")
    
    # Test forward pass
    batch_size = 16
    seq_len = 10
    x = torch.randn(batch_size, seq_len, 5)
    day_open = torch.randn(batch_size, 1)
    
    output = model(x, day_open)
    print(f"Input shape: {x.shape}")
    print(f"Day open shape: {day_open.shape}")
    print(f"Output shape: {output.shape}")
    
    assert output.shape == (batch_size, 1), "Output shape mismatch"
    print("✓ Test passed!")

