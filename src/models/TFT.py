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
    """ Gated Residual Network with Pre-Normalization (Google SOTA style) """

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super(GRN, self).__init__()
        self.ln = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.gate = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.ln(x)
        h = F.elu(self.fc1(h))
        h = self.fc2(h)
        h = self.dropout(h)
        g = torch.sigmoid(self.gate(x))
        return x + g * h


class VSN(nn.Module):
    """ Variable Selection Network: The 'Brain' that picks the best features """

    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super(VSN, self).__init__()
        self.feature_grns = nn.ModuleList([
            GRN(1, hidden_dim, hidden_dim, dropout) for _ in range(input_dim)
        ])
        self.flattened_grn = GRN(input_dim * hidden_dim, hidden_dim, input_dim, dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: [batch, time, input_dim]
        batch, time, dim = x.shape
        feature_outputs = []
        for i in range(dim):
            f_in = x[:, :, i:i + 1]  # Single feature
            feature_outputs.append(self.feature_grns[i](f_in))  # [batch, time, hidden]

        combined = torch.cat(feature_outputs, dim=-1)  # [batch, time, dim*hidden]
        weights = self.softmax(self.flattened_grn(combined)).unsqueeze(-1)  # [batch, time, dim, 1]

        stacked = torch.stack(feature_outputs, dim=2)  # [batch, time, dim, hidden]
        return torch.sum(weights * stacked, dim=2)  # Weighted context


class StockTFT(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=128, n_heads=8, output_dim=3, dropout=0.2, num_layers=1):
        super(StockTFT, self).__init__()
        self.vsn = VSN(input_dim, hidden_dim, dropout)

        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True)

        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim * 2, num_heads=n_heads,
                                         dropout=dropout, batch_first=True)

        self.post_attn_grn = GRN(hidden_dim * 2, hidden_dim, hidden_dim * 2, dropout=dropout)
        self.classifier = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        x = self.vsn(x)  # Drastic Improvement: Variable Selection
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.mha(lstm_out, lstm_out, lstm_out)
        x = self.post_attn_grn(attn_out)
        return self.classifier(x[:, -1, :])