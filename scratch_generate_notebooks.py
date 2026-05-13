import json
import os

def create_notebook(filename, cells_data):
    cells = []
    for cell_type, content in cells_data:
        # Split by newline and append newline to all except the last if needed
        # Actually standard nbformat likes a list of strings, each ending in \n
        lines = [line + '\n' for line in content.split('\n')]
        # Remove trailing newline from the very last string to avoid extra blank lines
        if lines:
            lines[-1] = lines[-1].rstrip('\n')
            
        cell = {
            "cell_type": cell_type,
            "metadata": {},
            "source": lines
        }
        if cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)
        
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.9.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)

# ==========================================
# 1. TRAINING NOTEBOOK
# ==========================================
training_md_1 = """# Deep Ensemble PINN Training

This notebook trains an ensemble of 5 identical Physics-Informed Neural Networks (PINNs) using 5 distinct random mathematical initializations.
This allows us to capture the AI's Epistemic Uncertainty by measuring the variance across the outputs of all 5 networks.
"""

training_code_setup = r"""import os
import random
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import joblib

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Load Data
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
    'CMRR': 'CMRR(dB)', 'class': 'Class', 'CLASS': 'Class'
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
CLASSIFICATION_TARGET = 'Class'

X_raw = df[FEATURE_COLUMNS].fillna(df[FEATURE_COLUMNS].mean())
y_reg_raw = df[REGRESSION_TARGETS].fillna(df[REGRESSION_TARGETS].mean())
y_class_raw = df[CLASSIFICATION_TARGET].fillna(df[CLASSIFICATION_TARGET].mode()[0])

scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X_raw)

scaler_y_reg = StandardScaler()
y_reg_scaled = scaler_y_reg.fit_transform(y_reg_raw)

le = LabelEncoder()
y_class_labels = le.fit_transform(y_class_raw)
n_classes = len(le.classes_)

# Save scalers for Inverse predictor
joblib.dump(scaler_X, 'scaler_X.pkl')
joblib.dump(scaler_y_reg, 'scaler_y_reg.pkl')
joblib.dump(le, 'label_encoder.pkl')

idx_all = np.arange(len(X_scaled))
train_val_idx, test_idx = train_test_split(idx_all, test_size=0.2, random_state=42, stratify=y_class_labels)
train_idx, val_idx = train_test_split(train_val_idx, test_size=0.2, random_state=42, stratify=y_class_labels[train_val_idx])

X_train = torch.tensor(X_scaled[train_idx], dtype=torch.float32).to(device)
X_val = torch.tensor(X_scaled[val_idx], dtype=torch.float32).to(device)
y_reg_train = torch.tensor(y_reg_scaled[train_idx], dtype=torch.float32).to(device)
y_reg_val = torch.tensor(y_reg_scaled[val_idx], dtype=torch.float32).to(device)
y_class_train = torch.tensor(y_class_labels[train_idx], dtype=torch.long).to(device)
y_class_val = torch.tensor(y_class_labels[val_idx], dtype=torch.long).to(device)

train_loader = DataLoader(TensorDataset(X_train, y_reg_train, y_class_train), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_reg_val, y_class_val), batch_size=64, shuffle=False)

print('Data loaded and preprocessed successfully.')
"""

training_code_model = r"""class PINN(nn.Module):
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

def physics_loss_normalized(x_tensor_scaled, y_pred_reg_scaled):
    device = x_tensor_scaled.device
    x_mean = torch.tensor(scaler_X.mean_, dtype=torch.float32, device=device)
    x_scale = torch.tensor(scaler_X.scale_, dtype=torch.float32, device=device)
    y_mean = torch.tensor(scaler_y_reg.mean_, dtype=torch.float32, device=device)
    y_scale = torch.tensor(scaler_y_reg.scale_, dtype=torch.float32, device=device)

    x_unscaled_tensor = x_tensor_scaled * x_scale + x_mean
    y_pred_unscaled_tensor = y_pred_reg_scaled * y_scale + y_mean

    feature_map = {name: x_unscaled_tensor[:, i] for i, name in enumerate(FEATURE_COLUMNS)}
    W12 = feature_map['W12(um)']
    W34 = feature_map['W34(um)']
    W6 = feature_map['W6(um)']
    Idc = feature_map['Idc(uA)']
    L = feature_map['Length(um)']
    Cc = feature_map['CC(pF)']

    gain_pred = y_pred_unscaled_tensor[:, REGRESSION_TARGETS.index('Gain(dB)')]
    bw_pred = y_pred_unscaled_tensor[:, REGRESSION_TARGETS.index('Bandwidth(Hz)')]
    gbw_pred = y_pred_unscaled_tensor[:, REGRESSION_TARGETS.index('GBW(MHz)')]
    pm_pred = y_pred_unscaled_tensor[:, REGRESSION_TARGETS.index('PM(degree)')]
    sr_pred = y_pred_unscaled_tensor[:, REGRESSION_TARGETS.index('SlewRate (V/us)')]
    power_pred = y_pred_unscaled_tensor[:, REGRESSION_TARGETS.index('Power(uW)')]
    cmrr_pred = y_pred_unscaled_tensor[:, REGRESSION_TARGETS.index('CMRR(dB)')]

    u_nCox, u_pCox = 343.98, 107.1
    lambda_n, lambda_p = 0.1, 0.2
    VDD = 1.8

    Id1_amps = (Idc / 2) * 1e-6
    Cc_farads = Cc * 1e-12

    gm12 = torch.sqrt(2 * u_nCox * (W12 / L) * (Idc / 2)) * 1e-6
    gm34 = torch.sqrt(2 * u_pCox * (W34 / L) * (Idc / 2)) * 1e-6
    gm6 = torch.sqrt(2 * u_nCox * (W6 / L) * Idc) * 1e-6
    gm_avg = (gm12 + gm34 + gm6) / 3
    ro12 = (1 / (lambda_n * (Idc / 2))) * 1e6
    ro34 = (1 / (lambda_p * (Idc / 2))) * 1e6
    ro6 = (1 / (lambda_n * Idc)) * 1e6
    ro7 = (1 / (lambda_n * Idc)) * 1e6

    gain_calc_db = 20 * torch.log10(((gm12 * (ro12 * ro34) / (ro12 + ro34)) * (gm6 * (ro6 * ro7) / (ro6 + ro7)) * 0.033) + 1e-8)

    C1, C2, C3 = 11.28, 0.133, 1e-6
    gain_linear = 10 ** (gain_pred / 20.0)
    pm_correction = C1 - (C2 * pm_pred)
    gbw_calc_mhz = gain_linear * bw_pred * pm_correction * C3

    sr_calc_vus = ((Id1_amps / Cc_farads) / 1e6) * 2.88
    power_calc_uw = (VDD * Idc) * 9.17
    cmrr_calc_db = 20 * torch.log10(((gm_avg / 1e-6) * 2.85) + 1e-8)

    loss_gain = F.mse_loss(gain_pred, gain_calc_db) / 6400.0
    loss_gbw = F.mse_loss(gbw_pred, gbw_calc_mhz) / 10000.0
    loss_sr = F.mse_loss(sr_pred, sr_calc_vus) / 400.0
    loss_power = F.mse_loss(power_pred, power_calc_uw) / 9000000.0
    loss_cmrr = F.mse_loss(cmrr_pred, cmrr_calc_db) / 10000.0

    return loss_gain + loss_gbw + loss_sr + loss_power + loss_cmrr
"""

training_code_loop = r"""def train_pinn_model(model, train_loader, val_loader, epochs=60, alpha_max=0.58, class_w=4.30):
    optimizer = optim.Adam(model.parameters(), lr=0.0008657)
    crit_reg = nn.MSELoss()
    crit_class = nn.CrossEntropyLoss()
    
    for epoch in range(1, epochs + 1):
        model.train()
        alpha = min(alpha_max, alpha_max * (epoch / 50)) if epoch < 50 else alpha_max

        for xb, yrb, ycb in train_loader:
            xb = xb.clone().detach().requires_grad_(True)
            optimizer.zero_grad()
            pr, pc = model(xb)

            loss_sup = crit_reg(pr, yrb) + (class_w * crit_class(pc, ycb))
            loss_phys = physics_loss_normalized(xb, pr)
            loss = (1 - alpha) * loss_sup + alpha * loss_phys
            loss.backward()
            optimizer.step()
    return model

# ---------------------------------------------------------
# DEEP ENSEMBLE TRAINING LOOP
# ---------------------------------------------------------
N_ENSEMBLE = 5
for ensemble_idx in range(1, N_ENSEMBLE + 1):
    seed = 42 + ensemble_idx
    
    # 1. Initialize distinct random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    print(f"\n--- Training Ensemble Model {ensemble_idx}/{N_ENSEMBLE} (Seed: {seed}) ---")
    
    # 2. Instantiate Model
    pinn = PINN(
        input_dim=len(FEATURE_COLUMNS),
        hidden_dims=[128, 128, 128, 128],
        n_reg_outputs=len(REGRESSION_TARGETS),
        n_classes=n_classes,
        dropout_rate=0.047,
    ).to(device)
    
    # 3. Train
    start_time = time.time()
    pinn = train_pinn_model(pinn, train_loader, val_loader, epochs=60, alpha_max=0.58, class_w=4.30)
    print(f"Training completed in {time.time() - start_time:.2f} seconds.")
    
    # 4. Save distinct weight file
    model_filepath = f'pinn_ens_{ensemble_idx}.pth'
    torch.save(pinn.state_dict(), model_filepath)
    print(f"Saved -> {model_filepath}")

print("\nAll 5 Ensembles Successfully Trained and Saved!")
"""

cells_training = [
    ("markdown", training_md_1),
    ("code", training_code_setup),
    ("code", training_code_model),
    ("code", training_code_loop)
]

# ==========================================
# 2. INVERSE NOTEBOOK
# ==========================================
inverse_md_1 = """# Deep Ensemble Inverse Predictor

This notebook resolves the Many-To-One problem of analog design by generating valid combinations of widths.
It uses an **Adam Multi-Start** algorithm generating 1,000 potential topologies.
To prevent the "Hallucination Loophole", we route these widths through an ensemble of 5 PINNs to penalize Epistemic uncertainty.
Finally, we apply physical photolithography grid-snapping post-predictions.
"""

inverse_code_setup = r"""import os
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

# 1. LOAD DATA & SCALERS
file_path = '../Data/FINAL_4CLASSES.csv'
df = pd.read_csv(file_path, encoding='utf-8', engine='python')
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

column_mapping = {'Gain(db)': 'Gain(dB)', 'Gain': 'Gain(dB)', 'gain': 'Gain(dB)'}
df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns}, inplace=True)

scaler_X = joblib.load('scaler_X.pkl')
scaler_y_reg = joblib.load('scaler_y_reg.pkl')
le = joblib.load('label_encoder.pkl')
n_classes = len(le.classes_)

# 2. DEFINE & LOAD ENSEMBLES
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

N_ENSEMBLE = 5
ensemble_models = nn.ModuleList()

for i in range(1, N_ENSEMBLE + 1):
    model = PINN(
        input_dim=len(FEATURE_COLUMNS),
        hidden_dims=[128, 128, 128, 128],
        n_reg_outputs=len(REGRESSION_TARGETS),
        n_classes=n_classes,
        dropout_rate=0.047,
    ).to(device)
    
    model.load_state_dict(torch.load(f'pinn_ens_{i}.pth', map_location=device))
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    ensemble_models.append(model)
    
print(f"Loaded {N_ENSEMBLE} pre-trained PINN models for Epistemic Uncertainty Estimation.")
"""

inverse_code_opt = r"""# 3. BATCH OPTIMIZATION TARGET
sample_row = df.sample(n=1, random_state=123).iloc[0]

target_metrics = {col: sample_row[col] for col in REGRESSION_TARGETS}
operating_conditions = {
    'Temperature(°)': sample_row['Temperature(°)'],
    'Idc(uA)': sample_row['Idc(uA)'],
    'Length(um)': sample_row['Length(um)'],
    'CC(pF)': sample_row['CC(pF)'],
    'CL(pF)': sample_row['CL(pF)']
}

target_array = np.array([[target_metrics[col] for col in REGRESSION_TARGETS]])
target_scaled = scaler_y_reg.transform(target_array)
target_tensor = torch.tensor(target_scaled, dtype=torch.float32).to(device)

width_names = ['W12(um)', 'W34(um)', 'W58(um)', 'W6(um)', 'W7(um)']
width_indices = [FEATURE_COLUMNS.index(col) for col in width_names]
fixed_indices = [FEATURE_COLUMNS.index(col) for col in operating_conditions.keys()]

mu_widths_1d = torch.tensor(scaler_X.mean_[width_indices], dtype=torch.float32, device=device)
std_widths_1d = torch.tensor(scaler_X.scale_[width_indices], dtype=torch.float32, device=device)

# Provide 1000 noisy starts
N_STARTS = 1000
noise_scale = 1.5
inv_widths_raw = (mu_widths_1d + torch.randn((N_STARTS, len(width_names)), device=device) * noise_scale).clone().detach().requires_grad_(True)

fake_dict = {**operating_conditions, **{k: 0 for k in width_names}}
initial_features_array = np.array([[fake_dict[col] for col in FEATURE_COLUMNS]])
initial_scaled = scaler_X.transform(initial_features_array)
fixed_scaled_vals = torch.tensor(initial_scaled[0, fixed_indices], dtype=torch.float32, device=device).unsqueeze(0).repeat(N_STARTS, 1)

# Inverse Optimizer
inv_epochs = 1500
inv_optimizer = optim.Adam([inv_widths_raw], lr=0.1)  
LAMBDA_UQ = 2.0  # Weight penalty for Epistemic Uncertainty 

print(f"\nTarget Performance to Meet:")
for k, v in target_metrics.items():
    print(f" - {k}: {v:.2f}")

print(f"\nStarted optimizing {N_STARTS} distinct geometries using Deep Ensemble UQ Guardrails...")

for epoch in range(1, inv_epochs + 1):
    inv_optimizer.zero_grad()
    
    widths_phys = F.softplus(inv_widths_raw)
    widths_scaled = (widths_phys - mu_widths_1d) / std_widths_1d
    
    full_scaled_input = torch.zeros((N_STARTS, len(FEATURE_COLUMNS)), dtype=torch.float32, device=device)
    full_scaled_input[:, width_indices] = widths_scaled
    full_scaled_input[:, fixed_indices] = fixed_scaled_vals
    
    # --- DEEP ENSEMBLE FORWARD PASS ---
    all_preds_reg = []
    for model in ensemble_models:
        p_reg, _ = model(full_scaled_input)
        all_preds_reg.append(p_reg)
        
    # Stack predictions: shape -> [N_ENSEMBLE, N_STARTS, N_OUTPUTS]
    stacked_preds = torch.stack(all_preds_reg)
    mean_preds = stacked_preds.mean(dim=0)
    var_preds = stacked_preds.var(dim=0, unbiased=False)
    
    # Loss 1: Accuracy MSE against targets (using the Mean of Ensembles)
    loss_mse_batch = F.mse_loss(mean_preds, target_tensor.expand_as(mean_preds), reduction='none').mean(dim=1)
    
    # Loss 2: Uncertainty Variance (Epistemic UQ Guardrail)
    loss_uq_batch = var_preds.mean(dim=1) 
    
    # Total combined loss
    total_batch_loss = loss_mse_batch + LAMBDA_UQ * loss_uq_batch
    
    loss = total_batch_loss.mean()
    loss.backward()
    inv_optimizer.step()
    
    if epoch % 300 == 0 or epoch == inv_epochs:
        print(f"Epoch {epoch:04d}/{inv_epochs} | Total Loss: {loss.item():.6f} | MSE: {loss_mse_batch.mean().item():.6f} | UQ Var: {loss_uq_batch.mean().item():.6f}")

final_losses = loss_mse_batch.detach().cpu().numpy()
final_widths_phys = F.softplus(inv_widths_raw).detach().cpu().numpy()
print("Optimization Complete.")
"""

inverse_code_eval = r"""# 4. GRID SNAPPING AND UQ BOUNDING
threshold = 0.05
success_mask = final_losses < threshold
valid_widths = final_widths_phys[success_mask]

if len(valid_widths) == 0:
    print("No solutions found under strictly specified targets.")
else:
    K = min(5, len(valid_widths))
    kmeans = KMeans(n_clusters=K, random_state=42).fit(valid_widths)
    
    best_representatives = []
    for cluster_id in range(K):
        cluster_indices = np.where(kmeans.labels_ == cluster_id)[0]
        # Sort by actual MSE
        best_idx_in_cluster = cluster_indices[np.argmin(final_losses[success_mask][cluster_indices])]
        best_representatives.append(best_idx_in_cluster)

    print("\n=======================================================")
    print(" TOP DIVERSE GEOMETRIES (GRID-SNAPPED TO 0.005um bounds)")
    print("=======================================================\n")
    
    for i, rep_idx in enumerate(best_representatives):
        w_values_cont = valid_widths[rep_idx]
        
        # MATHMATICAL GRID SNAPPING 
        # VLSI Grid is commonly 0.005um. We enforce rounding mapping to discrete geometries mathematically.
        grid_step = 0.005 
        w_values_snapped = np.round(w_values_cont / grid_step) * grid_step
        
        print(f"--- DESIGN OPTION #{i+1} ---")
        for j, col in enumerate(width_names):
            print(f"   > {col}: {w_values_snapped[j]:.3f}um")
            
        # Re-evaluate performance bounds with mapped grid limits
        best_w_scaled = (torch.tensor(w_values_snapped, dtype=torch.float32, device=device) - mu_widths_1d) / std_widths_1d
        best_full_scaled = torch.zeros((1, len(FEATURE_COLUMNS)), dtype=torch.float32, device=device)
        best_full_scaled[0, width_indices] = best_w_scaled
        best_full_scaled[0, fixed_indices] = fixed_scaled_vals[0]
        
        # Deep Ensemble Evaluation passing purely deterministic snapped layouts
        preds_unscaled = []
        for model in ensemble_models:
            with torch.no_grad():
                pred_eval, _ = model(best_full_scaled)
                preds_unscaled.append(scaler_y_reg.inverse_transform(pred_eval.cpu().numpy())[0])
                
        preds_unscaled = np.array(preds_unscaled)
        mean_perf = preds_unscaled.mean(axis=0)
        std_perf = preds_unscaled.std(axis=0)
        
        print("\n   [ PERFORMANCE PREDICTION ± UNCERTAINTY ]")
        for idx, col in enumerate(REGRESSION_TARGETS):
            target_v = target_metrics[col]
            mean_v = mean_perf[idx]
            uq_v = std_perf[idx] * 2  # 2 standard deviations (95% confidence bounds)
            
            # Formatting Output to explicitly state ± limits mathematically decoupled from Layout geometries
            err_pct = abs(mean_v - target_v) / (abs(target_v)+1e-8) * 100
            print(f"   - {col:16s} : Target={target_v:8.2f} | Out= {mean_v:8.2f} ± {uq_v:<5.2f}  (Epistemic: {uq_v:.2f})")
        print("\n" + "="*55 + "\n")
"""

cells_inverse = [
    ("markdown", inverse_md_1),
    ("code", inverse_code_setup),
    ("code", inverse_code_opt),
    ("code", inverse_code_eval)
]

create_notebook('c:/Users/LENOVO/Desktop/Physics Informed Neural Networks For EDA/Deep Ensemble/Deep_Ensemble_Training.ipynb', cells_training)
create_notebook('c:/Users/LENOVO/Desktop/Physics Informed Neural Networks For EDA/Deep Ensemble/Deep_Ensemble_Inverse.ipynb', cells_inverse)
print("Notebooks Generated Successfully.")
