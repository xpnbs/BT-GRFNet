# -*- coding: utf-8 -*-
"""
BiLSTM + GCN Regression (5-fold CV)
- 使用真实电网邻接矩阵
- BiLSTM 序列建模 + GCN 图特征提取
- Early Stopping + ReduceLROnPlateau
- 回归 + 分箱分类准确率
"""

import os
import random
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ===================== Config =====================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ===================== Paths =====================
data_dir = r"/电网运行状态（10000）"
labels_path = r"/标签1.csv"
adj_path = r"/邻接矩阵.csv"


# ===================== Training hyperparams =====================
n_splits = 5
batch_size = 64
num_epochs = 100
learning_rate = 5e-4
weight_decay = 5e-5
grad_clip = 2.0
early_stop_patience = 10
scheduler_patience = 5
scheduler_factor = 0.5

# ===================== Model hyperparams =====================
lstm_hidden = 128
lstm_layers = 2
gcn_hidden = 64
dropout = 0.2

# ===================== Class mapping =====================
def map_to_class_scalar(v: float) -> int:
    if v <= 0.2: return 0
    elif v <= 0.4: return 1
    elif v <= 0.6: return 2
    elif v <= 0.8: return 3
    else: return 4

# ===================== Load data =====================
labels_df = pd.read_csv(labels_path)
labels_dict = dict(zip(labels_df.iloc[:,0].astype(str), labels_df.iloc[:,1]))

X_list, y_list = [], []
for i in range(1, 10001):
    fp = os.path.join(data_dir, f"运行状态{i}.csv")
    if not os.path.exists(fp):
        continue
    df = pd.read_csv(fp)
    X_list.append(df.iloc[:,1:11].apply(pd.to_numeric, errors="coerce").fillna(0).values)
    y_list.append(labels_dict[str(i)])

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.float32).reshape(-1,1)
N, seq_len, feat_dim = X.shape
print(f"Loaded X: {X.shape}, y: {y.shape}")

# ===================== Standardize =====================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.reshape(-1, feat_dim)).reshape(N, seq_len, feat_dim)
y_strat = np.array([map_to_class_scalar(v) for v in y.flatten()])

# ===================== Load adjacency =====================
adj_df = pd.read_csv(adj_path, header=None)
adj_matrix = torch.tensor(adj_df.values, dtype=torch.float32)
adj_matrix.fill_diagonal_(1)
assert adj_matrix.shape[0] == seq_len

# ===================== GCN Module =====================
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        # Graph Convolution: X' = A X W
        support = self.fc(x)
        out = torch.matmul(adj, support)
        return F.relu(out)

class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.gcn1 = GCNLayer(in_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        x = self.gcn1(x, adj)
        x = self.dropout(x)
        x = self.gcn2(x, adj)
        return x  # output shape: (B, N, hidden_dim)

# ===================== BiLSTM + GCN Regressor =====================
class BiLSTMGCNRegressor(nn.Module):
    def __init__(self, input_dim, adj_matrix):
        super().__init__()
        self.bilstm = nn.LSTM(input_dim, lstm_hidden, num_layers=lstm_layers,
                              batch_first=True, bidirectional=True, dropout=dropout)
        self.lstm_proj = nn.Linear(2*lstm_hidden, lstm_hidden)

        self.gcn = GCN(input_dim, gcn_hidden, dropout)
        self.register_buffer("adj", adj_matrix)

        self.fusion_norm = nn.LayerNorm(lstm_hidden + gcn_hidden)
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden + gcn_hidden, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128,1)
        )

    def forward(self, x):
        # BiLSTM branch
        lstm_out, _ = self.bilstm(x)
        lstm_feat = self.lstm_proj(lstm_out)
        lstm_feat = lstm_feat.mean(dim=1)

        # GCN branch
        gcn_feat = self.gcn(x, self.adj)
        gcn_feat = gcn_feat.mean(dim=1)

        # Fusion
        fused = torch.cat([lstm_feat, gcn_feat], dim=-1)
        fused = self.fusion_norm(fused)
        return self.head(fused).squeeze(-1)

# ===================== Train / Eval =====================
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device).squeeze(-1)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        total += loss.item() * xb.size(0)
    return total / len(loader.dataset)

def evaluate(model, loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            preds.append(model(xb.to(device)).cpu().numpy())
            trues.append(yb.numpy().squeeze(-1))
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    mse = mean_squared_error(trues, preds)
    return preds, trues, mse, np.sqrt(mse), mean_absolute_error(trues, preds), r2_score(trues, preds)

# ===================== 5-Fold Cross-validation =====================
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
all_metrics = []

for fold, (tr, va) in enumerate(skf.split(X_scaled, y_strat), 1):
    print(f"\n========== Fold {fold} ==========")

    train_loader = DataLoader(TensorDataset(torch.tensor(X_scaled[tr]), torch.tensor(y[tr])),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_scaled[va]), torch.tensor(y[va])),
                            batch_size=batch_size)

    model = BiLSTMGCNRegressor(feat_dim, adj_matrix).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=scheduler_factor, patience=scheduler_patience)
    criterion = nn.MSELoss()

    best_rmse, patience = 1e9, 0
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        _, _, mse, rmse, mae, r2 = evaluate(model, val_loader)
        scheduler.step(mse)
        print(f"Epoch {epoch+1:03d} | TrainLoss={train_loss:.6f} | RMSE={rmse:.6f}")

        if rmse < best_rmse:
            best_rmse = rmse
            patience = 0
            torch.save(model.state_dict(), f"best_fold{fold}.pt")
        else:
            patience += 1
            if patience >= early_stop_patience:
                break

    model.load_state_dict(torch.load(f"best_fold{fold}.pt"))
    preds, trues, mse, rmse, mae, r2 = evaluate(model, val_loader)
    acc = accuracy_score([map_to_class_scalar(v) for v in trues],
                         [map_to_class_scalar(v) for v in preds])
    all_metrics.append({"Fold": fold, "RMSE": rmse, "MAE": mae, "R2": r2, "ClassAcc": acc})

# ===================== Summary =====================
df = pd.DataFrame(all_metrics)
df.loc["Average"] = df.mean(numeric_only=True)
df.to_csv("BiLSTM_GCN_cv_metrics.csv", index=False)
print("\n===== Final Results =====")
print(df)
