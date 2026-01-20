import torch
import torch.nn as nn
import torch.nn.functional as F


# --- STRATEGY A: Standard Weighted Cross-Entropy ---
class WeightedCELoss(nn.Module):
    def __init__(self, alpha=None):
        super(WeightedCELoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(weight=alpha)

    def forward(self, logits, targets, epoch=None):
        return self.ce(logits, targets)


# --- STRATEGY B: The "Bold Sniper" Loss ---
class SniperLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.5, directional_weight=1.2, exploration_bonus=0.02):
        super(SniperLoss, self).__init__()
        self.gamma = gamma
        self.directional_weight = directional_weight
        self.exploration_bonus = exploration_bonus
        self.ce = nn.CrossEntropyLoss(weight=alpha, reduction='none')

    def forward(self, logits, targets, epoch=None):
        # 1. Focal Component: Focuses on 'hard' breakout signals
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()

        # 2. Symmetric Directional Penalty
        probs = F.softmax(logits, dim=1)
        pred_labels = torch.argmax(probs, dim=1)
        opposite_mask = ((targets == 2) & (pred_labels == 0)) | ((targets == 0) & (pred_labels == 2))

        warmup = min(1.0, epoch / 20.0) if epoch is not None else 1.0
        directional_penalty = opposite_mask.float().mean() * self.directional_weight * warmup

        # 3. Entropy Bonus (Anti-Coward Metric)
        entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=1).mean()

        return focal_loss + directional_penalty - (self.exploration_bonus * entropy)


# --- MODEL ARCHITECTURE ---

class GRN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.gate = nn.Linear(hidden_dim, output_dim)
        self.ln = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = F.elu(self.fc1(x))
        h = self.fc2(h)
        h = self.dropout(h)
        g = torch.sigmoid(self.gate(F.elu(self.fc1(x))))
        return self.ln(x + g * h)


class StockTFT(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=128, n_heads=8, output_dim=3, dropout=0.2, num_layers=1):
        super(StockTFT, self).__init__()
        self.vsn = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim * 2, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.post_attn_grn = GRN(hidden_dim * 2, hidden_dim, hidden_dim * 2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = F.elu(self.vsn(x))
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.mha(lstm_out, lstm_out, lstm_out)
        x = self.post_attn_grn(attn_out)
        return self.classifier(torch.mean(x, dim=1))