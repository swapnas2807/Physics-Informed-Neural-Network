import os

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

def train_pinn_model(model, train_loader, val_loader, epochs=60, alpha_max=0.58, class_w=4.30):

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

