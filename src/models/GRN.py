import torch
import torch.nn as nn
import torch.nn.functional as F


class TFTLoss(nn.Module):
    def __init__(self, alpha=None, gamma=1.5, directional_weight=0.6):
        super(TFTLoss, self).__init__()
        # alpha should be the class_weights tensor
        self.alpha = alpha
        self.gamma = gamma
        self.dw = directional_weight

    def forward(self, inputs, targets):
        # 1. Calculate Weighted Cross Entropy FIRST
        # This provides the base 'pressure' to balance classes
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)

        # 2. Apply Focal Scaling
        # This focuses the model on the 'hard-to-classify' windows
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        # 3. Directional Penalty
        # Heavily punishes flipping a 'Buy' signal into a 'Sell' signal
        preds = inputs.argmax(dim=1)
        directional_mask = ((preds == 0) & (targets == 2)) | ((preds == 2) & (targets == 0))

        final_loss = focal_loss.clone()
        final_loss[directional_mask] *= (1 + self.dw)

        return final_loss.mean()


class GRN(nn.Module):
    """Gated Residual Network used for post-attention refinement."""

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super(GRN, self).__init__()
        self.ln = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.gate = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x):
        h = self.ln(x)
        h = F.elu(self.fc1(h))
        h = self.fc2(h)
        h = self.dropout(h)
        g = torch.sigmoid(self.gate(x))
        return self.skip(x) + g * h


class StockTFT(nn.Module):
    """
    Reconstructed 'Wide-Shallow' Success Model.
    Architecture: Vectorized VSN -> Bi-LSTM -> Multi-Head Attention -> GRN -> Classifier
    """

    def __init__(self, input_dim=6, hidden_dim=128, n_heads=8, output_dim=3, dropout=0.1):
        super(StockTFT, self).__init__()

        # 1. Vectorized VSN: Projects 6 features (OHLC + Vol + $Vol) into hidden space
        self.vsn = nn.Linear(input_dim, hidden_dim)

        # 2. Bi-Directional LSTM: Single layer depth (Shallow) but wide
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)

        # 3. Multi-Head Attention: 8 heads, embed_dim = hidden_dim * 2
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim * 2, num_heads=n_heads, batch_first=True)

        # 4. Post-Attention Refinement
        self.post_attn_grn = GRN(hidden_dim * 2, hidden_dim, hidden_dim * 2, dropout=dropout)

        # 5. Final Classification Head
        self.classifier = nn.Linear(hidden_dim * 2, output_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0.0)

    def forward(self, x):
        x = self.vsn(x)
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.mha(lstm_out, lstm_out, lstm_out)
        x = self.post_attn_grn(attn_out)
        return self.classifier(x[:, -1, :])