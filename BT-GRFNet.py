
import os
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score

import torch
import torch.nn as nn
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
d_model = 128
nhead = 8
num_layers = 2
dim_feedforward = 512
dropout = 0.2
lstm_hidden = 128
lstm_layers = 2

# ===================== Class mapping =====================
def map_to_class_scalar(v: float) -> int:
    if v <= 0.2: return 0
    elif v <= 0.4: return 1
    elif v <= 0.6: return 2
    elif v <= 0.8: return 3
    else: return 4

# ===================== Load data =====================
labels_df = pd.read_csv(labels_path)
labels_dict = dict(zip(labels_df.iloc[:, 0].astype(str), labels_df.iloc[:, 1]))

X_list, y_list = [], []
for i in range(1, 10001):
    fp = os.path.join(data_dir, f"运行状态{i}.csv")
    if not os.path.exists(fp):
        continue
    df = pd.read_csv(fp)
    X_list.append(df.iloc[:, 1:11].apply(pd.to_numeric, errors="coerce").fillna(0).values)
    y_list.append(labels_dict[str(i)])

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.float32).reshape(-1, 1)
N, seq_len, feat_dim = X.shape

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.reshape(-1, feat_dim)).reshape(N, seq_len, feat_dim)
y_strat = np.array([map_to_class_scalar(v) for v in y.flatten()])

# ===================== Modules =====================
# ----- Transformer-only -----
class TransformerOnlyRegressor(nn.Module):
    def __init__(self, seq_len, input_dim):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.input_proj(x) + self.pos_embedding
        x = self.transformer(x)
        x = self.norm(x)
        feat = x.mean(dim=1)
        return self.head(feat).squeeze(-1)

# ----- BiLSTM-only -----
class BiLSTMRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.bilstm = nn.LSTM(input_dim, lstm_hidden, num_layers=lstm_layers,
                              batch_first=True, bidirectional=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(2*lstm_hidden, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128,1)
        )
    def forward(self, x):
        feat, _ = self.bilstm(x)
        feat = feat.mean(dim=1)
        return self.head(feat).squeeze(-1)

# ----- Gated Residual Fusion -----
class GatedResidualFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim*2, dim),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(dim)
    def forward(self, x1, x2):
        alpha = self.gate(torch.cat([x1, x2], dim=-1))
        fused = alpha*x1 + (1-alpha)*x2 + x1 + x2
        return self.norm(fused)

# ----- Full Model (with fusion) -----
class TransformerBiLSTMRegressor(nn.Module):
    def __init__(self, input_dim, fusion=True):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, d_model)*0.02)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.bilstm = nn.LSTM(d_model, lstm_hidden, num_layers=lstm_layers,
                              batch_first=True, bidirectional=True, dropout=dropout)
        self.lstm_proj = nn.Linear(2*lstm_hidden, d_model)

        self.fusion_enabled = fusion
        if fusion:
            self.fusion = GatedResidualFusion(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model//2,1)
        )

    def forward(self, x):
        x = self.input_proj(x) + self.pos_emb
        trans_feat = self.transformer(x)
        lstm_out, _ = self.bilstm(x)
        lstm_feat = self.lstm_proj(lstm_out)
        if self.fusion_enabled:
            fused = self.fusion(trans_feat, lstm_feat)
        else:
            fused = trans_feat + lstm_feat  # simple sum if no fusion
        pooled = fused.mean(dim=1)
        return self.head(pooled).squeeze(-1)

# ===================== Train / Eval =====================
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device).squeeze(-1)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
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

# ===================== 4. Cross-validation for 4 models =====================
models_to_run = {
    "Full Model": lambda: TransformerBiLSTMRegressor(feat_dim, fusion=True),
    "wo Fusion": lambda: TransformerBiLSTMRegressor(feat_dim, fusion=False),
    "Transformer-only": lambda: TransformerOnlyRegressor(seq_len, feat_dim),
    "BiLSTM-only": lambda: BiLSTMRegressor(feat_dim)
}

for model_name, model_class in models_to_run.items():
    print(f"\n===== Experiment: {model_name} =====")
    out_dir_model = f"{model_name.replace(' ','_')}_results"
    model_dir = os.path.join(out_dir_model, "models")
    os.makedirs(model_dir, exist_ok=True)
    all_metrics = []

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for fold, (tr, va) in enumerate(skf.split(X_scaled, y_strat),1):
        print(f"\n--- Fold {fold} ---")
        train_loader = DataLoader(TensorDataset(torch.tensor(X_scaled[tr]), torch.tensor(y[tr])),
                                  batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(torch.tensor(X_scaled[va]), torch.tensor(y[va])),
                                batch_size=batch_size)

        model = model_class().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=scheduler_factor, patience=scheduler_patience
        )
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
                torch.save(model.state_dict(), f"{model_dir}/best_fold{fold}.pt")
            else:
                patience += 1
                if patience >= early_stop_patience:
                    break

        model.load_state_dict(torch.load(f"{model_dir}/best_fold{fold}.pt"))
        preds, trues, _, rmse, mae, r2 = evaluate(model, val_loader)
        acc = accuracy_score([map_to_class_scalar(v) for v in trues],
                             [map_to_class_scalar(v) for v in preds])
        all_metrics.append({"Fold": fold, "RMSE": rmse, "MAE": mae, "R2": r2, "ClassAcc": acc})

    df = pd.DataFrame(all_metrics)
    df.loc["Average"] = df.mean(numeric_only=True)
    df.to_csv(os.path.join(out_dir_model,"cv_metrics.csv"), index=False)
    print(df)
