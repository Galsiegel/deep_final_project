"""
Classification model for stock direction prediction.

Predicts price movement direction (5 classes).
"""

import torch
import torch.nn as nn
from models.gru_model import GRUEncoder


class StockClassificationModel(nn.Module):
    """
    GRU-based model for stock direction classification.
    
    Predicts one of 5 classes:
        0: Large Down (< -2%)
        1: Small Down (-2% to -0.5%)
        2: Neutral (-0.5% to +0.5%)
        3: Small Up (+0.5% to +2%)
        4: Large Up (> +2%)
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int = 5,
        dropout_gru: float = 0.2,
        dropout_fc: float = 0.3,
        fc_hidden_size: int = 64
    ):
        """
        Initialize classification model.
        
        Args:
            input_size: Number of input features (5 for OHLCV)
            hidden_size: GRU hidden size
            num_layers: Number of GRU layers
            num_classes: Number of output classes
            dropout_gru: Dropout between GRU layers
            dropout_fc: Dropout in FC layers
            fc_hidden_size: Size of FC layer before output
        """
        super().__init__()
        
        self.num_classes = num_classes
        
        # Shared GRU encoder
        self.encoder = GRUEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout_gru=dropout_gru,
            dropout_fc=dropout_fc,
            fc_hidden_size=fc_hidden_size
        )
        
        # Classification head: predict class logits
        # Input: GRU features + opening price (fc_hidden_size + 1)
        self.head = nn.Linear(fc_hidden_size + 1, num_classes)
    
    def forward(self, x: torch.Tensor, day_open: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, seq_len, input_size] - historical OHLCV
            day_open: Opening price tensor [batch, 1] - day i+1's opening
            
        Returns:
            Class logits [batch, num_classes]
        """
        # Encode sequence
        features = self.encoder(x)  # [batch, fc_hidden_size]
        
        # Concatenate GRU features with opening price
        combined = torch.cat([features, day_open], dim=1)  # [batch, fc_hidden_size + 1]
        
        # Predict class logits
        logits = self.head(combined)
        
        return logits
    
    def predict_class(self, x: torch.Tensor, day_open: torch.Tensor) -> torch.Tensor:
        """
        Predict class labels (argmax of logits).
        
        Args:
            x: Input tensor [batch, seq_len, input_size]
            day_open: Opening price tensor [batch, 1]
            
        Returns:
            Predicted classes [batch]
        """
        logits = self.forward(x, day_open)
        return torch.argmax(logits, dim=1)
    
    def predict_proba(self, x: torch.Tensor, day_open: torch.Tensor) -> torch.Tensor:
        """
        Predict class probabilities (softmax of logits).
        
        Args:
            x: Input tensor [batch, seq_len, input_size]
            day_open: Opening price tensor [batch, 1]
            
        Returns:
            Class probabilities [batch, num_classes]
        """
        logits = self.forward(x, day_open)
        return torch.softmax(logits, dim=1)
    
    def get_num_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick test
    print("Testing StockClassificationModel...")
    
    model = StockClassificationModel(
        input_size=5,
        hidden_size=128,
        num_layers=2,
        num_classes=5,
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
    
    logits = model(x, day_open)
    probs = model.predict_proba(x, day_open)
    preds = model.predict_class(x, day_open)
    
    print(f"Input shape: {x.shape}")
    print(f"Day open shape: {day_open.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Probs shape: {probs.shape}")
    print(f"Predictions shape: {preds.shape}")
    
    assert logits.shape == (batch_size, 5), "Logits shape mismatch"
    assert probs.shape == (batch_size, 5), "Probs shape mismatch"
    assert preds.shape == (batch_size,), "Predictions shape mismatch"
    assert torch.allclose(probs.sum(dim=1), torch.ones(batch_size)), "Probs don't sum to 1"
    
    print("✓ All tests passed!")

