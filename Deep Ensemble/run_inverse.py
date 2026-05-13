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



# 1. LOAD DATA & SCALERS

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

# 3. BATCH OPTIMIZATION TARGET

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

# 4. GRID SNAPPING AND UQ BOUNDING

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

