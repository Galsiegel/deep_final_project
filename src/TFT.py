"""
src/TFT.py
Implements the Temporal Fusion Transformer.
Contains modular implementations of GRN, VSN, and Directional Focal Loss.
Designed for financial time-series classification with static stock embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np


# --- 1. CORE BUILDING BLOCKS ---

class GRN(nn.Module):
    """
    Gated Residual Network (GRN).
    Provides adaptive non-linear processing with a skip connection and gating mechanism.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(output_dim, output_dim)
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.ln = nn.LayerNorm(output_dim)

    def forward(self, x):
        h = F.elu(self.fc1(x))
        h = self.fc2(h)
        h = self.dropout(h)
        g = torch.sigmoid(self.gate(h))
        h = g * h
        out = self.skip(x) + h
        return self.ln(out)


class VSN(nn.Module):
    """
    Variable Selection Network (VSN).
    Automatically identifies relevant features at each time step using GRNs and Softmax weights.
    """

    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.feature_grns = nn.ModuleList([GRN(1, hidden_dim, hidden_dim, dropout) for _ in range(input_dim)])

        self.selector = nn.Sequential(
            nn.Linear(input_dim * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x shape: [Batch, Sequence, Features]
        feats = [self.feature_grns[i](x[:, :, i:i + 1]) for i in range(x.shape[-1])]
        combined = torch.cat(feats, dim=-1)

        logits = self.selector(combined)
        weights = self.softmax(logits).unsqueeze(-1)  # [Batch, Seq, Feat, 1]

        stacked = torch.stack(feats, dim=2)  # [Batch, Seq, Feat, Hidden]
        out = (weights * stacked).sum(dim=2)

        # Store weights for post-training interpretability
        self.last_weights = weights.detach()
        return out


# --- 2. MAIN ARCHITECTURE ---

class StockTFT(nn.Module):
    """
    Stock-Specific Temporal Fusion Transformer.
    Combines VSN for feature selection, Embeddings for stock identity,
    and LSTM + Attention for temporal patterns.
    """

    def __init__(self,
                 input_dim=13,
                 hidden_dim=64,
                 n_heads=8,
                 output_dim=3,
                 dropout=0.2,
                 num_stocks=48,
                 stock_emb_dim=769,
                 num_layers=1):
        super().__init__()

        # 1. Feature Selection
        self.vsn = VSN(input_dim, hidden_dim, dropout)

        # 2. Static Context Integration
        self.stock_emb = nn.Embedding(num_stocks, stock_emb_dim)
        pre_dim = hidden_dim + stock_emb_dim

        self.pre_lstm_ln = nn.LayerNorm(pre_dim)
        self.pre_lstm_proj = nn.Linear(pre_dim, hidden_dim)

        # 3. Temporal Processing (LSTM + Attention)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True)

        # LSTM is BiDirectional, so embed_dim is hidden_dim * 2
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim * 2, num_heads=n_heads,
                                         dropout=dropout, batch_first=True)

        # 4. Post-Processing & Head
        self.post_attn_grn = GRN(hidden_dim * 2, hidden_dim, hidden_dim * 2, dropout=dropout)
        self.classifier = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x, sid):
        """
        Args:
            x: Technical/News features [Batch, Seq_Len, Features]
            sid: Static Stock IDs [Batch]
        """
        # Variable Selection
        x = self.vsn(x)

        # Inject Stock Embedding across sequence
        e = self.stock_emb(sid).unsqueeze(1).expand(-1, x.size(1), -1)
        x = torch.cat([x, e], dim=-1)

        # LSTM Temporal encoding
        x = self.pre_lstm_ln(x)
        x = self.pre_lstm_proj(x)
        lstm_out, _ = self.lstm(x)

        # Global attention mechanism
        attn_out, _ = self.mha(lstm_out, lstm_out, lstm_out)

        # Final gated residual skip and classification
        x = self.post_attn_grn(attn_out)
        return self.classifier(x[:, -1, :])  # Predict based on the last sequence step


# --- 3. LOSS & UTILITIES ---

class CustomFocalLoss(nn.Module):
    """
    Focal Loss with Directional Penalty.
    Designed to heavily penalize 'opposite' predictions (e.g., predicting Buy when it is a Sell).
    - Higher values of gamma increase the focus on difficult samples
    - Higher values of directional-weights (dir_weight) prioritizes avoiding opposite-direction trades over
      general classification accuracy
    """

    def __init__(self, alpha=None, gamma=1.6, dir_weight=0.6):
        super().__init__()
        self.gamma = gamma
        self.dir_weight = dir_weight
        self.ce = nn.CrossEntropyLoss(weight=alpha, reduction='none')

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)

        # 1. Standard Focal Loss component
        focal = ((1 - pt) ** self.gamma * ce_loss).mean()

        # 2. Directional Penalty (penalize extreme misses)
        probs = F.softmax(logits, dim=1)
        pen_target_0 = (targets == 0).to(torch.float32)
        pen_target_2 = (targets == 2).to(torch.float32)

        penalty = (probs[:, 2] * pen_target_0 + probs[:, 0] * pen_target_2).mean()

        return (1 - self.dir_weight) * focal + self.dir_weight * penalty


def initialize_weights(m):
    """Xavier and Orthogonal initialization for stability in deep LSTMs."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None: nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)


# --- 4. DATASET ---

class MasterStockDataset(Dataset):
    """Dataset class for loading dataset.pt files with metadata preservation."""
    def __init__(self, path):
        self.samples = torch.load(path, weights_only=False)
        self.X = torch.tensor(np.array([s['x'] for s in self.samples]), dtype=torch.float32)
        self.y = torch.tensor(np.array([s['y'] for s in self.samples]), dtype=torch.long)
        self.sid = torch.tensor(np.array([s['static_id'] for s in self.samples]), dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.sid[idx], self.y[idx]

    def get_class_weights(self):
        counts = np.bincount(self.y.numpy(), minlength=3)
        return torch.tensor(len(self.y)/(3*np.maximum(counts, 1)), dtype=torch.float32)