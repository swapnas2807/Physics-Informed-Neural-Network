import os
import time
import warnings
import joblib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


# ---------------------------
# 1. LOAD DATA & SCALERS & PINN
# ---------------------------
file_path = '../Data/FINAL_4CLASSES.csv'
df = pd.read_csv(file_path, encoding='utf-8', engine='python')

column_mapping = {
    'Gain(db)': 'Gain(dB)', 'Gain': 'Gain(dB)', 'gain': 'Gain(dB)',
    'Bandwidth': 'Bandwidth(Hz)', 'bandwidth': 'Bandwidth(Hz)',
    'GBW': 'GBW(MHz)', 'gbw': 'GBW(MHz)',
    'Power': 'Power(uW)', 'power': 'Power(uW)',
    'PM': 'PM(degree)', 'PhaseMargin': 'PM(degree)',
    'GM': 'GM(dB)',
    'PSRR': 'PSRR(dB)',
    'SlewRate': 'SlewRate (V/us)', 'SlewRate(V/µs)': 'SlewRate (V/us)',
    'CMRR': 'CMRR(dB)'
}
df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns}, inplace=True)

df['Idc(uA)'] = 130.0
df['Length(um)'] = 0.18
df['CL(pF)'] = 10.0
df['CC(pF)'] = 55.0

FEATURE_COLUMNS = [
    'Temperature(°)', 'W12(um)', 'W34(um)', 'W58(um)', 'W6(um)', 'W7(um)',
    'Idc(uA)', 'Length(um)', 'CC(pF)', 'CL(pF)'
]
REGRESSION_TARGETS = [
    'Gain(dB)', 'Bandwidth(Hz)', 'GBW(MHz)', 'Power(uW)', 'PM(degree)',
    'GM(dB)', 'PSRR(dB)', 'SlewRate (V/us)', 'CMRR(dB)'
]

scaler_X = joblib.load('scaler_X.pkl')
scaler_y_reg = joblib.load('scaler_y_reg.pkl')
le = joblib.load('label_encoder.pkl')
n_classes = len(le.classes_)
print("Loaded Scalers.")

class PINN(nn.Module):
    def __init__(self, input_dim, hidden_dims, n_reg_outputs, n_classes, dropout_rate):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.regression_head = nn.Linear(prev_dim, n_reg_outputs)
        self.classification_head = nn.Linear(prev_dim, n_classes)

    def forward(self, x):
        shared = self.backbone(x)
        return self.regression_head(shared), self.classification_head(shared)

pinn = PINN(
    input_dim=len(FEATURE_COLUMNS),
    hidden_dims=[128, 128, 128, 128],
    n_reg_outputs=len(REGRESSION_TARGETS),
    n_classes=n_classes, # Based on dataset classes
    dropout_rate=0.047,
).to(device)

pinn.load_state_dict(torch.load('pinn_model.pth', map_location=device))
pinn.eval()
for param in pinn.parameters():
    param.requires_grad = False
print("Loaded pre-trained PINN model.")


# ---------------------------
# 2. BATCH OPTIMIZATION SETUP
# ---------------------------
# Grab a random sample for our target reference
sample_row = df.sample(n=1, random_state=123).iloc[0]

target_metrics = {col: sample_row[col] for col in REGRESSION_TARGETS}
operating_conditions = {
    'Temperature(°)': sample_row['Temperature(°)'],
    'Idc(uA)': sample_row['Idc(uA)'],
    'Length(um)': sample_row['Length(um)'],
    'CC(pF)': sample_row['CC(pF)'],
    'CL(pF)': sample_row['CL(pF)']
}

print(f"Target Performance to Meet:")
for k, v in target_metrics.items():
    print(f" - {k}: {v:.2f}")

target_array = np.array([[target_metrics[col] for col in REGRESSION_TARGETS]])
target_scaled = scaler_y_reg.transform(target_array)
# Change the target tensor to broadcast automatically across N guesses
target_tensor = torch.tensor(target_scaled, dtype=torch.float32).to(device)

width_names = ['W12(um)', 'W34(um)', 'W58(um)', 'W6(um)', 'W7(um)']
width_indices = [FEATURE_COLUMNS.index(col) for col in width_names]
fixed_indices = [FEATURE_COLUMNS.index(col) for col in operating_conditions.keys()]

mu_widths_1d = torch.tensor(scaler_X.mean_[width_indices], dtype=torch.float32, device=device)
std_widths_1d = torch.tensor(scaler_X.scale_[width_indices], dtype=torch.float32, device=device)

# --- THE MAGIC (N = 1000 PARALLEL RUNS) ---
N_STARTS = 1000
# We initialize raw parameter starting at mu_widths, but we inject a large Gaussian noise scaling matrix
noise_scale = 1.5
inv_widths_raw = (mu_widths_1d + torch.randn((N_STARTS, len(width_names)), device=device) * noise_scale).clone().detach().requires_grad_(True)

fake_dict = {**operating_conditions, **{k: 0 for k in width_names}}
initial_features_array = np.array([[fake_dict[col] for col in FEATURE_COLUMNS]])
initial_scaled = scaler_X.transform(initial_features_array)
# Broadcast fixed scaled vals to match batch size
fixed_scaled_vals = torch.tensor(initial_scaled[0, fixed_indices], dtype=torch.float32, device=device).unsqueeze(0).repeat(N_STARTS, 1)

# Inverse Optimizer configuration
inv_epochs = 1500
inv_optimizer = optim.Adam([inv_widths_raw], lr=0.1)  

print(f"\nStarted optimizing {N_STARTS} distinct geometries simultaneously...")
for epoch in range(1, inv_epochs + 1):
    inv_optimizer.zero_grad()
    
    # 1. Constrain physical width to be POSITIVE 
    widths_phys = F.softplus(inv_widths_raw)
    
    # 2. Scale back the widths internally
    widths_scaled = (widths_phys - mu_widths_1d) / std_widths_1d
    
    full_scaled_input = torch.zeros((N_STARTS, len(FEATURE_COLUMNS)), dtype=torch.float32, device=device)
    full_scaled_input[:, width_indices] = widths_scaled
    full_scaled_input[:, fixed_indices] = fixed_scaled_vals
    
    pred_reg, _ = pinn(full_scaled_input)
    
    # We use MSELoss with reduction='none' so we can see which batch items succeed independently
    # However we backprop the mean over all items.
    loss_batch = F.mse_loss(pred_reg, target_tensor.expand_as(pred_reg), reduction='none').mean(dim=1)
    loss = loss_batch.mean()
    loss.backward()
    inv_optimizer.step()
    
    if epoch % 500 == 0 or epoch == inv_epochs:
        print(f"Search Epoch {epoch:04d}/{inv_epochs} | Average Global MSE: {loss.item():.6f}")

final_losses = loss_batch.detach().cpu().numpy()
final_widths_phys = F.softplus(inv_widths_raw).detach().cpu().numpy()
print("Optimization Complete.")


# Filter only successful solutions
threshold = 0.05
success_mask = final_losses < threshold
valid_widths = final_widths_phys[success_mask]
valid_losses = final_losses[success_mask]

print(f"Total valid solutions found (MSE < {threshold}): {len(valid_widths)} / {N_STARTS}")

if len(valid_widths) == 0:
    print("No solutions found under the strict threshold. Try a higher threshold or more epochs.")
else:
    # Cluster the valid geometries to present distinct design choices
    K = min(5, len(valid_widths))  # Maximum 5 distinct families presented
    kmeans = KMeans(n_clusters=K, random_state=42).fit(valid_widths)
    
    best_representatives = []
    for cluster_id in range(K):
        # Indices of widths in this cluster
        cluster_indices = np.where(kmeans.labels_ == cluster_id)[0]
        # Find the one within this cluster with the absolute lowest physics loss
        best_idx_in_cluster = cluster_indices[np.argmin(valid_losses[cluster_indices])]
        best_representatives.append(best_idx_in_cluster)
        
    print(f"\n============================")
    print(f" TOP {K} DIVERSE GEOMETRIES:")
    print(f"============================\n")
    
    ground_truth = {k: sample_row[k] for k in width_names}
    print("Ground Truth in Dataset:", ", ".join([f"{k}: {v:.2f}" for k,v in ground_truth.items()]))
    print("-"*60)
    
    for i, rep_idx in enumerate(best_representatives):
        print(f"\n--- DESIGN OPTION #{i+1} (Loss: {valid_losses[rep_idx]:.5f}) ---")
        w_values = valid_widths[rep_idx]
        for j, col in enumerate(width_names):
            print(f"   > {col}: {w_values[j]:.4f}")
            
    # Verify Performance of Option #1
    best_opt = best_representatives[0]
    best_w_scaled = (torch.tensor(valid_widths[best_opt], dtype=torch.float32, device=device) - mu_widths_1d) / std_widths_1d
    best_full_scaled = torch.zeros((1, len(FEATURE_COLUMNS)), dtype=torch.float32, device=device)
    best_full_scaled[0, width_indices] = best_w_scaled
    best_full_scaled[0, fixed_indices] = fixed_scaled_vals[0]
    
    pinn.eval()
    with torch.no_grad():
        pred_eval, _ = pinn(best_full_scaled)
        pred_unscaled = scaler_y_reg.inverse_transform(pred_eval.cpu().numpy())[0]
        
    print("\n\nPERFORMANCE CHECK FOR DESIGN OPTION #1:")
    for i, col in enumerate(REGRESSION_TARGETS):
        t_val = target_metrics[col]
        p_val = pred_unscaled[i]
        err_pct = abs(p_val - t_val) / (abs(t_val)+1e-8) * 100
        print(f"   > {col}: Target={t_val:.2f} | Predicted={p_val:.2f} | Error: {err_pct:.2f}%")


