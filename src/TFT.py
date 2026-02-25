"""
src/TFT.py
Implements the Temporal Fusion Transformer (Dual-Path Architecture).
Contains modular implementations of GRN, VSN, EmbeddingBranch, and Directional Focal Loss.
Designed for financial time-series classification with:
  - Path 1: Technical+Sentiment features -> VSN -> feature selection
  - Path 2: News Embedding [768] -> dedicated layers -> dimensionality reduction
  - Path 3: Stock Embedding -> projection layers
All paths merge before LSTM for temporal processing.
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


# --- 2. EMBEDDING BRANCH ---

class EmbeddingBranch(nn.Module):
    """
    Processes high-dimensional news embeddings through dedicated layers
    to produce a reduced-dimension representation.
    Input:  [Batch, Seq, 768]
    Output: [Batch, Seq, hidden_dim]
    """

    def __init__(self, embedding_dim=768, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        """
        x: [Batch, Seq, 768]
        returns: [Batch, Seq, hidden_dim]
        """
        batch, seq, _ = x.shape

        # Reshape for BatchNorm: [Batch*Seq, Features]
        x = x.reshape(batch * seq, -1)

        # Layer 1: 768 -> 256
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout(x)

        # Layer 2: 256 -> hidden_dim
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.activation(x)

        # Reshape back: [Batch, Seq, hidden_dim]
        x = x.reshape(batch, seq, -1)
        return x


# --- 3. MAIN ARCHITECTURE ---

class StockTFT(nn.Module):
    """
    Dual-Path Temporal Fusion Transformer.
    Path 1: Technical+Sentiment [M, 17] -> VSN -> [M, hidden_dim]
    Path 2: News Embedding [M, 768] -> EmbeddingBranch (768->256->hidden_dim)
    Path 3: Stock Embedding -> Projection (stock_emb_dim -> 128 -> hidden_dim)
    All paths merge before LSTM for temporal processing.
    """

    def __init__(self,
                 tech_input_dim=17,      # 13 technical + 4 sentiment
                 embedding_dim=768,       # news embedding dimension
                 hidden_dim=64,
                 n_heads=8,
                 output_dim=3,
                 dropout=0.2,
                 num_stocks=48,
                 stock_emb_dim=256,       # learnable stock embedding size
                 num_layers=1,
                 use_news=True):          # flag to disable news path
        super().__init__()

        self.use_news = use_news

        # Path 1: Technical + Sentiment through VSN
        self.vsn = VSN(tech_input_dim, hidden_dim, dropout)

        # Path 2: News Embedding processing (768 -> 256 -> hidden_dim)
        if use_news:
            self.embedding_branch = EmbeddingBranch(embedding_dim, hidden_dim, dropout)
            merge_dim = hidden_dim * 3  # VSN + EmbeddingBranch + StockProjection
        else:
            merge_dim = hidden_dim * 2  # VSN + StockProjection only

        # Path 3: Stock Embedding with projection layers
        self.stock_emb = nn.Embedding(num_stocks, stock_emb_dim)
        self.stock_proj = nn.Sequential(
            nn.Linear(stock_emb_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # Pre-LSTM fusion
        self.pre_lstm_ln = nn.LayerNorm(merge_dim)
        self.pre_lstm_proj = nn.Linear(merge_dim, hidden_dim)

        # Temporal Processing (LSTM + Attention)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True)

        # LSTM is BiDirectional, so embed_dim is hidden_dim * 2
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim * 2, num_heads=n_heads,
                                         dropout=dropout, batch_first=True)

        # Post-Processing & Head
        self.post_attn_grn = GRN(hidden_dim * 2, hidden_dim, hidden_dim * 2, dropout=dropout)
        self.classifier = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x_tech, x_emb, sid):
        """
        Args:
            x_tech: Technical+Sentiment features [Batch, Seq_Len, 17]
            x_emb:  News embeddings [Batch, Seq_Len, 768]
            sid:    Static Stock IDs [Batch]
        """
        # Path 1: Variable Selection on technical + sentiment features
        x_main = self.vsn(x_tech)  # [Batch, Seq, hidden_dim]

        # Path 2: Process news embeddings (768 -> 256 -> hidden_dim)
        if self.use_news:
            x_news = self.embedding_branch(x_emb)  # [Batch, Seq, hidden_dim]

        # Path 3: Process stock embedding and expand across sequence
        e = self.stock_emb(sid)    # [Batch, stock_emb_dim]
        e = self.stock_proj(e)     # [Batch, hidden_dim]
        e = e.unsqueeze(1).expand(-1, x_tech.size(1), -1)  # [Batch, Seq, hidden_dim]

        # Merge all paths
        if self.use_news:
            x = torch.cat([x_main, x_news, e], dim=-1)  # [Batch, Seq, hidden_dim*3]
        else:
            x = torch.cat([x_main, e], dim=-1)           # [Batch, Seq, hidden_dim*2]

        # Pre-LSTM projection
        x = self.pre_lstm_ln(x)
        x = self.pre_lstm_proj(x)  # [Batch, Seq, hidden_dim]

        # LSTM Temporal encoding
        lstm_out, _ = self.lstm(x)

        # Global attention mechanism
        attn_out, _ = self.mha(lstm_out, lstm_out, lstm_out)

        # Final gated residual skip and classification
        x = self.post_attn_grn(attn_out)
        return self.classifier(x[:, -1, :])  # Predict based on the last sequence step


# --- 4. LOSS & UTILITIES ---

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


# --- 5. DATASET ---

class MasterStockDataset(Dataset):
    """
    Dataset class supporting both legacy and dual-path formats.
    Legacy:    sample has 'x' key          -> returns (x, dummy_emb, sid, y)
    Dual-path: sample has 'x_tech' key     -> returns (x_tech, x_emb, sid, y)
    """

    def __init__(self, path):
        self.samples = torch.load(path, weights_only=False)

        # Detect dataset format
        if 'x_tech' in self.samples[0]:
            self.is_dual_path = True
            self.X_tech = torch.tensor(
                np.array([s['x_tech'] for s in self.samples]),
                dtype=torch.float32
            )
            self.X_emb = torch.tensor(
                np.array([s['x_emb'] for s in self.samples]),
                dtype=torch.float32
            )
        else:
            self.is_dual_path = False
            self.X = torch.tensor(
                np.array([s['x'] for s in self.samples]),
                dtype=torch.float32
            )

        self.y = torch.tensor(
            np.array([s['y'] for s in self.samples]),
            dtype=torch.long
        )
        self.sid = torch.tensor(
            np.array([s['static_id'] for s in self.samples]),
            dtype=torch.long
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if self.is_dual_path:
            return self.X_tech[idx], self.X_emb[idx], self.sid[idx], self.y[idx]
        else:
            # Legacy: return dummy embedding tensor so the interface stays uniform
            seq_len = self.X.shape[1]
            dummy_emb = torch.zeros(seq_len, 768)
            return self.X[idx], dummy_emb, self.sid[idx], self.y[idx]

    def get_class_weights(self):
        counts = np.bincount(self.y.numpy(), minlength=3)
        return torch.tensor(len(self.y) / (3 * np.maximum(counts, 1)), dtype=torch.float32)
