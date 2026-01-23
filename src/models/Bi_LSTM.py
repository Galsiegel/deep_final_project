import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedLinearUnit(nn.Module):
    """Splits input to create a gating mechanism for non-linear processing."""

    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, input_dim * 2)

    def forward(self, x):
        x = self.fc(x)
        return F.glu(x, dim=-1)


class GatedResidualNetwork(nn.Module):
    """Primary building block of TFT for flexible non-linear mapping."""

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.glu = GatedLinearUnit(output_dim)
        self.ln = nn.LayerNorm(output_dim)
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = F.elu(self.fc1(x))
        h = self.fc2(h)
        h = self.dropout(h)
        h = self.glu(h)
        return self.ln(self.skip(x) + h)


class VariableSelectionNetwork(nn.Module):
    """Assigns importance weights to your 6 OHLCV+ features."""

    def __init__(self, num_features, hidden_dim, dropout=0.1):
        super().__init__()
        self.feature_grns = nn.ModuleList([
            GatedResidualNetwork(1, hidden_dim, hidden_dim, dropout) for _ in range(num_features)
        ])
        self.flattened_grn = GatedResidualNetwork(num_features, hidden_dim, num_features, dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: [batch, seq_len, 6]
        batch, seq, n_feat = x.shape
        feature_outputs = []
        for i in range(n_feat):
            feature_outputs.append(self.feature_grns[i](x[:, :, i:i + 1]))

        # Calculate selection weights per time step
        flattened_x = x.view(-1, n_feat)
        weights = self.softmax(self.flattened_grn(flattened_x)).view(batch, seq, n_feat, 1)

        combined = torch.stack(feature_outputs, dim=-2)
        return (weights * combined).sum(dim=-2)


class TFTQuantileLoss(nn.Module):
    """
    Implements the Pinball Loss for 0.1, 0.5, and 0.9 quantiles.
    This creates a probabilistic 'cone' around the prediction.
    """

    def __init__(self, quantiles=[0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, targets):
        targets = targets.view(-1, 1).float()
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = targets - preds[:, i:i + 1]
            # Probabilistic penalty: encourages the model to center the median
            # while respecting the distribution bounds.
            losses.append(torch.max((q - 1) * errors, q * errors))
        return torch.mean(torch.stack(losses).sum(dim=0))


class StockTFT(nn.Module):
    """
    Precise Google TFT Architecture for Multi-Horizon Regression.
    """

    def __init__(self, input_dim=6, hidden_dim=128, n_heads=8, dropout=0.1):
        super().__init__()
        # 1. Variable Selection: Weighting OHLC vs Volume
        self.vsn = VariableSelectionNetwork(input_dim, hidden_dim, dropout)

        # 2. Temporal Processing: Wide-Shallow Bi-LSTM core
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)

        # 3. Post-LSTM Refinement (Static Enrichment substitution)
        self.post_lstm_grn = GatedResidualNetwork(hidden_dim * 2, hidden_dim, hidden_dim * 2, dropout)

        # 4. Attention for global feature fusion
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim * 2, num_heads=n_heads, batch_first=True)

        # 5. Probabilistic Quantile Output Head
        self.quantile_head = nn.Sequential(
            GatedResidualNetwork(hidden_dim * 2, hidden_dim, hidden_dim, dropout),
            nn.Linear(hidden_dim, 3)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # Xavier Uniform ensures variance is preserved across gates
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        # vsn_out: [batch, seq, hidden]
        vsn_out = self.vsn(x)

        # lstm_out: [batch, seq, hidden * 2]
        lstm_out, _ = self.lstm(vsn_out)

        # Apply gating and residual connection to temporal features
        enriched = self.post_lstm_grn(lstm_out)

        # Attention layer to weigh important past days
        attn_out, _ = self.mha(enriched, enriched, enriched)

        # Final forecast for the last time step
        # Predicts [q10, q50, q90] which define the return distribution probability.
        return self.quantile_head(attn_out[:, -1, :])