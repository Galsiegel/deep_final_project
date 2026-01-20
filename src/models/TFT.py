import torch
import torch.nn as nn
import torch.nn.functional as F


class TFTLoss(nn.Module):
    def __init__(self, alpha=None, gamma=1.5, directional_weight=0.6):
        super(TFTLoss, self).__init__()
        self.gamma = gamma
        self.directional_weight = directional_weight
        self.ce = nn.CrossEntropyLoss(weight=alpha, reduction='none')

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()

        probs = F.softmax(logits, dim=1)
        pred_labels = torch.argmax(probs, dim=1)
        opposite_mask = ((targets == 2) & (pred_labels == 0)) | ((targets == 0) & (pred_labels == 2))
        directional_penalty = opposite_mask.float().mean() * self.directional_weight

        return focal_loss + directional_penalty


class GRN(nn.Module):
    """ Gated Residual Network with Pre-Normalization """

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super(GRN, self).__init__()
        self.ln = nn.LayerNorm(input_dim)  # Applied FIRST
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.gate = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Pre-Norm path
        h = self.ln(x)
        h = F.elu(self.fc1(h))
        h = self.fc2(h)
        h = self.dropout(h)

        # Gating
        g = torch.sigmoid(self.gate(x))
        return x + g * h  # Residual connection is "clean"


class StockTFT(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=128, n_heads=8, output_dim=3, dropout=0.2, num_layers=1):
        super(StockTFT, self).__init__()
        self.vsn = nn.Linear(input_dim, hidden_dim)

        # Bi-LSTM for temporal context
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True)

        # Multi-Head Attention
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim * 2, num_heads=n_heads,
                                         dropout=dropout, batch_first=True)

        # SOTA Stability: Pre-Norm GRN for final classification
        self.post_attn_grn = GRN(hidden_dim * 2, hidden_dim, hidden_dim * 2, dropout=dropout)
        self.classifier = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # 1. Variable Selection (Simple version)
        x = F.elu(self.vsn(x))

        # 2. Temporal Processing
        lstm_out, _ = self.lstm(x)

        # 3. Attention
        attn_out, _ = self.mha(lstm_out, lstm_out, lstm_out)

        # 4. Gated Residual Connection (with Pre-Norm inside)
        x = self.post_attn_grn(attn_out)

        # 5. Global Temporal Pooling (Taking the last timestep)
        x = x[:, -1, :]
        return self.classifier(x)