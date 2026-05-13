
# --- CELL ---

# ==========================================================
# Strict reproducibility setup (run this first)
# ==========================================================
import os
import random
import numpy as np
import torch

SEED = 42

# Environment flags for deterministic CUDA behavior (must be set early).
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# Seed all major RNGs.
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

# Force deterministic algorithms where possible.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

try:
    torch.use_deterministic_algorithms(True)
    print('Deterministic mode enabled successfully.')
except Exception as exc:
    print(f'Warning: full deterministic mode could not be enabled: {exc}')

print(f'Seed fixed to: {SEED}')
print('Reproducibility cell executed. Now run remaining cells in order.')
# --- END CELL ---

import os
import time
import random
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.multioutput import MultiOutputRegressor

warnings.filterwarnings('ignore')

# ==========================================================
# 1) Reproducibility + global configuration
# ==========================================================
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seeds(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# ==========================================================
# 2) Data loading and preprocessing
# ==========================================================
file_path = '../Data/FINAL_4CLASSES.csv'
if not os.path.exists(file_path):
    raise FileNotFoundError(f'Dataset not found at: {file_path}')

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
    'CMRR': 'CMRR(dB)',
    'class': 'Class', 'CLASS': 'Class'
}
df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns}, inplace=True)

# --- STRATIFIED DATASET REDUCTION --- 
FRACTION = 0.2 # Try 20% of dataset
# Use grouping to uniformly sample across all classes
if 'Class' in df.columns:
    df = df.groupby('Class', group_keys=False).apply(lambda x: x.sample(frac=FRACTION, random_state=42))
else:
    df = df.sample(frac=FRACTION, random_state=42)
df = df.reset_index(drop=True)
print(f'Training on {FRACTION*100}% of data. Total samples: {len(df)}')

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
KEY_METRICS = ['Gain(dB)', 'Bandwidth(Hz)', 'GBW(MHz)', 'Power(uW)']

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
print('Data loaded and preprocessed successfully.')

# ==========================================================
# 3) Model definitions + physics loss
# ==========================================================
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

class StandardMLP(nn.Module):
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

def get_model_complexity(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def physics_loss_normalized(
    x_tensor_scaled,
    y_pred_reg_scaled,
    scaler_X_obj,
    scaler_y_reg_obj,
    feature_names,
    regression_target_names,
):
    dev = x_tensor_scaled.device
    x_mean = torch.tensor(scaler_X_obj.mean_, dtype=torch.float32, device=dev)
    x_scale = torch.tensor(scaler_X_obj.scale_, dtype=torch.float32, device=dev)
    y_mean = torch.tensor(scaler_y_reg_obj.mean_, dtype=torch.float32, device=dev)
    y_scale = torch.tensor(scaler_y_reg_obj.scale_, dtype=torch.float32, device=dev)

    x_unscaled = x_tensor_scaled * x_scale + x_mean
    y_pred_unscaled = y_pred_reg_scaled * y_scale + y_mean

    fmap = {name: x_unscaled[:, i] for i, name in enumerate(feature_names)}
    W12, W34, W6 = fmap['W12(um)'], fmap['W34(um)'], fmap['W6(um)']
    Idc, L, Cc = fmap['Idc(uA)'], fmap['Length(um)'], fmap['CC(pF)']

    gain_pred = y_pred_unscaled[:, regression_target_names.index('Gain(dB)')]
    bw_pred = y_pred_unscaled[:, regression_target_names.index('Bandwidth(Hz)')]
    gbw_pred = y_pred_unscaled[:, regression_target_names.index('GBW(MHz)')]
    pm_pred = y_pred_unscaled[:, regression_target_names.index('PM(degree)')]
    sr_pred = y_pred_unscaled[:, regression_target_names.index('SlewRate (V/us)')]
    power_pred = y_pred_unscaled[:, regression_target_names.index('Power(uW)')]
    cmrr_pred = y_pred_unscaled[:, regression_target_names.index('CMRR(dB)')]

    u_nCox, u_pCox = 343.98, 107.1
    lambda_n, lambda_p = 0.1, 0.2
    VDD = 1.8
    Id1_amps, Cc_farads = (Idc / 2) * 1e-6, Cc * 1e-12

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
    pm_correction = (C1 - (C2 * pm_pred))
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

def train_dl_model(model, train_loader, val_loader, epochs=50, is_pinn=False):
    optimizer = optim.Adam(model.parameters(), lr=0.0008657)
    crit_reg = nn.MSELoss()
    crit_class = nn.CrossEntropyLoss()

    alpha_max = 0.58 if is_pinn else 0.0
    class_w = 4.30
    history_rows = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0
        alpha = min(alpha_max, alpha_max * (epoch / 50)) if epoch < 50 else alpha_max

        for xb, yrb, ycb in train_loader:
            if is_pinn:
                xb.requires_grad_(True)
            optimizer.zero_grad()
            pr, pc = model(xb)

            loss_sup = crit_reg(pr, yrb) + (class_w * crit_class(pc, ycb))
            loss_phys = 0.0
            if is_pinn:
                loss_phys = physics_loss_normalized(
                    xb, pr, scaler_X, scaler_y_reg, FEATURE_COLUMNS, REGRESSION_TARGETS
                )

            loss = (1 - alpha) * loss_sup + alpha * loss_phys if is_pinn else loss_sup
            loss.backward()
            optimizer.step()

            train_loss += float(loss.item())
            pred_labels = torch.argmax(pc.data, dim=1)
            total_train += ycb.size(0)
            correct_train += int((pred_labels == ycb).sum().item())

        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for xb, yrb, ycb in val_loader:
                pr, pc = model(xb)
                loss_v = crit_reg(pr, yrb) + (class_w * crit_class(pc, ycb))
                val_loss += float(loss_v.item())
                pred_labels = torch.argmax(pc.data, dim=1)
                total_val += ycb.size(0)
                correct_val += int((pred_labels == ycb).sum().item())

        history_rows.append({
            'Epoch': epoch,
            'Train Loss': train_loss / len(train_loader) if len(train_loader) > 0 else 0,
            'Val Loss': val_loss / len(val_loader) if len(val_loader) > 0 else 0,
            'Train Accuracy': correct_train / total_train if total_train > 0 else 0,
            'Val Accuracy': correct_val / total_val if total_val > 0 else 0,
        })

    return model, pd.DataFrame(history_rows)

# ==========================================================
# 4) Training (Train/Test Split without K-Fold)
# ==========================================================
results_reg_rows = []
results_class_rows = []
per_target_rows = []
physics_rows = []
history_frames = []
all_fold_predictions = []

total_pinn_training_time = 0.0
pinn_complexity = 0

C1, C2, C3 = 11.28, 0.133, 1e-6

print('Training with an 80/20 train/test split...')

X_train_fold, X_test_fold, y_reg_train_fold, y_reg_test_fold, y_class_train_fold, y_class_test_fold = train_test_split(
    X_scaled, y_reg_scaled, y_class_labels, test_size=0.2, random_state=42, stratify=y_class_labels
)

X_train_t = torch.tensor(X_train_fold, dtype=torch.float32, device=device)
y_reg_train_t = torch.tensor(y_reg_train_fold, dtype=torch.float32, device=device)
y_class_train_t = torch.tensor(y_class_train_fold, dtype=torch.long, device=device)

X_test_t = torch.tensor(X_test_fold, dtype=torch.float32, device=device)
y_reg_test_t = torch.tensor(y_reg_test_fold, dtype=torch.float32, device=device)
y_class_test_t = torch.tensor(y_class_test_fold, dtype=torch.long, device=device)

dl_train = DataLoader(TensorDataset(X_train_t, y_reg_train_t, y_class_train_t), batch_size=64, shuffle=True)
dl_test = DataLoader(TensorDataset(X_test_t, y_reg_test_t, y_class_test_t), batch_size=64, shuffle=False)

# PINN
pinn = PINN(X_train_fold.shape[1], [128, 128, 128, 128], len(REGRESSION_TARGETS), n_classes, 0.047).to(device)
pinn_complexity = get_model_complexity(pinn)

t0 = time.time()
pinn, pinn_hist_df = train_dl_model(pinn, dl_train, dl_test, epochs=50, is_pinn=True)
total_pinn_training_time += (time.time() - t0)

pinn.eval()
with torch.no_grad():
    pr, pc = pinn(X_test_t)
    pred_reg_pinn = scaler_y_reg.inverse_transform(pr.cpu().numpy())
    pred_class_pinn = torch.argmax(pc, dim=1).cpu().numpy()

# Standard MLP
mlp = StandardMLP(X_train_fold.shape[1], [128, 128, 128, 128], len(REGRESSION_TARGETS), n_classes, 0.047).to(device)
mlp, mlp_hist_df = train_dl_model(mlp, dl_train, dl_test, epochs=50, is_pinn=False)

mlp.eval()
with torch.no_grad():
    pr, pc = mlp(X_test_t)
    pred_reg_mlp = scaler_y_reg.inverse_transform(pr.cpu().numpy())
    pred_class_mlp = torch.argmax(pc, dim=1).cpu().numpy()

# Random Forest
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
y_reg_train_raw = scaler_y_reg.inverse_transform(y_reg_train_fold)
rf_reg.fit(X_train_fold, y_reg_train_raw)
pred_reg_rf = rf_reg.predict(X_test_fold)

rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_clf.fit(X_train_fold, y_class_train_fold)
pred_class_rf = rf_clf.predict(X_test_fold)

# SVM
svm_reg = MultiOutputRegressor(SVR(kernel='rbf'))
svm_reg.fit(X_train_fold, y_reg_train_fold)
pred_reg_svm = scaler_y_reg.inverse_transform(svm_reg.predict(X_test_fold))

svm_clf = SVC(kernel='rbf')
svm_clf.fit(X_train_fold, y_class_train_fold)
pred_class_svm = svm_clf.predict(X_test_fold)

y_true_reg = scaler_y_reg.inverse_transform(y_reg_test_fold)
y_true_class = y_class_test_fold

preds = {
    'PINN': (pred_reg_pinn, pred_class_pinn),
    'Standard MLP': (pred_reg_mlp, pred_class_mlp),
    'Random Forest': (pred_reg_rf, pred_class_rf),
    'SVM': (pred_reg_svm, pred_class_svm),
    'y_true_reg': y_true_reg,
    'y_true_class': y_true_class,
}

for model_name, hist_df in [('PINN', pinn_hist_df), ('Standard MLP', mlp_hist_df)]:
    temp_hist = hist_df.copy()
    temp_hist['Model'] = model_name
    history_frames.append(temp_hist)

for model_name in ['PINN', 'Standard MLP', 'Random Forest', 'SVM']:
    pred_reg, pred_class = preds[model_name]

    # Overall fold-level metrics
    results_reg_rows.append({
        'Model': model_name,
        'R2': r2_score(y_true_reg, pred_reg),
        'RMSE': np.sqrt(mean_squared_error(y_true_reg, pred_reg)),
        'MAE': mean_absolute_error(y_true_reg, pred_reg),
        'MAPE': mean_absolute_percentage_error(y_true_reg, pred_reg),
    })

    results_class_rows.append({
        'Model': model_name,
        'Accuracy': accuracy_score(y_true_class, pred_class),
        'Precision': precision_score(y_true_class, pred_class, average='weighted', zero_division=0),
        'Recall': recall_score(y_true_class, pred_class, average='weighted', zero_division=0),
        'F1': f1_score(y_true_class, pred_class, average='weighted', zero_division=0),
    })

    # Per-target fold-level metrics
    for idx, target_name in enumerate(REGRESSION_TARGETS):
        yt = y_true_reg[:, idx]
        yp = pred_reg[:, idx]
        per_target_rows.append({
            'Model': model_name,
            'Target': target_name,
            'R2': r2_score(yt, yp),
            'RMSE': np.sqrt(mean_squared_error(yt, yp)),
            'MAE': mean_absolute_error(yt, yp),
            'MAPE': mean_absolute_percentage_error(yt, yp),
        })

    # Physics consistency + violation (GBW relation)
    idx_gain = REGRESSION_TARGETS.index('Gain(dB)')
    idx_bw = REGRESSION_TARGETS.index('Bandwidth(Hz)')
    idx_gbw = REGRESSION_TARGETS.index('GBW(MHz)')
    idx_pm = REGRESSION_TARGETS.index('PM(degree)')

    p_gain = pred_reg[:, idx_gain]
    p_bw = pred_reg[:, idx_bw]
    p_gbw = pred_reg[:, idx_gbw]
    p_pm = pred_reg[:, idx_pm]

    gain_lin = 10 ** (p_gain / 20.0)
    pm_corr = (C1 - (C2 * p_pm))
    gbw_calc = gain_lin * p_bw * pm_corr * C3

    relative_violation = np.abs((p_gbw - gbw_calc) / (gbw_calc + 1e-8))

    physics_rows.append({
        'Model': model_name,
        'Consistency_R2': r2_score(p_gbw, gbw_calc),
        'Consistency_MAPE': mean_absolute_percentage_error(gbw_calc, p_gbw),
        'Mean_Violation': np.mean(relative_violation),
        'Std_Violation': np.std(relative_violation),
        'Max_Violation': np.max(relative_violation),
        'Within_10pct': np.mean(relative_violation <= 0.10) * 100.0,
    })

# ==========================================================
# 5) Artifacts creation
# ==========================================================

df_reg_summary = pd.DataFrame(results_reg_rows)
df_class_summary = pd.DataFrame(results_class_rows)
df_target_summary = pd.DataFrame(per_target_rows)
df_physics_summary = pd.DataFrame(physics_rows)

print('\nTraining complete.')
print(f'PINN trainable parameters: {pinn_complexity:,}')
print(f'Total PINN training time: {total_pinn_training_time:.2f} s')
print('Artifacts are ready for tables.')
\n
# --- END CELL ---

# ==========================================================
# Hyperparameter summary used in this experiment
# ==========================================================
hyperparams = {
    'Global': {
        'Seed': 42,
        'Device': str(device),
        'Data file': file_path,
        'Feature count': len(FEATURE_COLUMNS),
        'Regression target count': len(REGRESSION_TARGETS),
        'Classification classes': n_classes,
    },
    'Cross-validation': {
        'K folds': 'None (Train/Test Split 80/20)',
    },
    'Deep learning (PINN + Standard MLP)': {
        'Hidden layers': [128, 128, 128, 128],
        'Dropout rate': 0.047,
        'Epochs': 50,
        'Batch size': 64,
        'Optimizer': 'Adam',
        'Learning rate': 0.0008657,
        'Regression loss': 'MSELoss',
        'Classification loss': 'CrossEntropyLoss',
        'Classification weight (class_w)': 4.30,
        'PINN alpha_max': 0.58,
        'PINN alpha schedule': 'linear warmup to alpha_max until epoch 50',
    },
    'Random Forest': {
        'n_estimators': 100,
        'random_state': 42,
        'n_jobs': -1,
    },
    'SVM': {
        'SVR kernel': 'rbf',
        'SVC kernel': 'rbf',
        'Multi-output strategy': 'MultiOutputRegressor(SVR)',
    },
    'Physics constants (GBW relation)': {
        'C1': 11.28,
        'C2': 0.133,
        'C3': 1e-6,
    },
}

print('=' * 95)
print('HYPERPARAMETERS USED IN THIS NOTEBOOK')
print('=' * 95)
for section, values in hyperparams.items():
    print(f'\n[{section}]')
    for k, v in values.items():
        print(f' - {k}: {v}')
\n
# --- END CELL ---

import pandas as pd

# ==========================================================
# Table utilities
# ==========================================================
def format_table(df_summary, value_columns, sort_by='Model'):
    df_out = df_summary.copy()
    for col in value_columns:
        df_out[col] = df_out[col].apply(lambda x: f"{x:.4f}")
    keep_cols = [c for c in ['Model', 'Target'] if c in df_out.columns] + value_columns
    df_out = df_out[keep_cols]
    if sort_by in df_out.columns:
        df_out = df_out.sort_values(sort_by).reset_index(drop=True)
    return df_out

print('=' * 95)
print('TABLE 1: REGRESSION PERFORMANCE')
print('=' * 95)
tbl_reg = format_table(df_reg_summary, ['R2', 'RMSE', 'MAE', 'MAPE'])
print(tbl_reg.to_csv(index=False, sep='\t'))

print('=' * 95)
print('TABLE 2: CLASSIFICATION PERFORMANCE')
print('=' * 95)
tbl_class = format_table(df_class_summary, ['Accuracy', 'Precision', 'Recall', 'F1'])
print(tbl_class.to_csv(index=False, sep='\t'))

print('=' * 95)
print('TABLE 3: PER-TARGET REGRESSION PERFORMANCE')
print('=' * 95)
tbl_target = format_table(df_target_summary, ['R2', 'RMSE', 'MAE', 'MAPE'], sort_by='Target')
print(tbl_target.to_csv(index=False, sep='\t'))

print('=' * 95)
print('TABLE 4: PHYSICS VIOLATION METRICS')
print('=' * 95)
tbl_physics = df_physics_summary[['Model']].copy()
tbl_physics['Mean Violation'] = df_physics_summary['Mean_Violation'].round(4)
tbl_physics['Std. Dev'] = df_physics_summary['Std_Violation'].round(4)
tbl_physics['Max. Violation'] = df_physics_summary['Max_Violation'].round(4)
tbl_physics['% Samples within ±10%'] = df_physics_summary['Within_10pct'].round(2)
print(tbl_physics.to_csv(index=False, sep='\t'))

print('=' * 95)
print('TABLE 5: R2 ACROSS KEY TARGETS FOR EACH MODEL')
print('=' * 95)
df_key_r2 = df_target_summary[df_target_summary['Target'].isin(KEY_METRICS)][['Model', 'Target', 'R2']].copy()
df_key_r2['R2'] = df_key_r2['R2'].apply(lambda x: f"{x:.4f}")
df_pivot_r2 = df_key_r2.pivot(index='Model', columns='Target', values='R2')[KEY_METRICS]
df_pivot_r2.columns = [f'{col} R2' for col in df_pivot_r2.columns]
df_pivot_r2 = df_pivot_r2.reset_index()
print(df_pivot_r2.to_csv(index=False, sep='\t'))

print('=' * 95)
print('TABLE 6: MAE ACROSS KEY TARGETS FOR EACH MODEL')
print('=' * 95)
df_key_mae = df_target_summary[df_target_summary['Target'].isin(KEY_METRICS)][['Model', 'Target', 'MAE']].copy()
df_key_mae['MAE'] = df_key_mae['MAE'].apply(lambda x: f"{x:.4f}")
df_pivot_mae = df_key_mae.pivot(index='Model', columns='Target', values='MAE')[KEY_METRICS]
df_pivot_mae.columns = [f'{col} MAE' for col in df_pivot_mae.columns]
df_pivot_mae = df_pivot_mae.reset_index()
print(df_pivot_mae.to_csv(index=False, sep='\t'))

print('=' * 95)
print('TABLE 7: RMSE ACROSS KEY TARGETS FOR EACH MODEL')
print('=' * 95)
df_key_rmse = df_target_summary[df_target_summary['Target'].isin(KEY_METRICS)][['Model', 'Target', 'RMSE']].copy()
df_key_rmse['RMSE'] = df_key_rmse['RMSE'].apply(lambda x: f"{x:.4f}")
df_pivot_rmse = df_key_rmse.pivot(index='Model', columns='Target', values='RMSE')[KEY_METRICS]
df_pivot_rmse.columns = [f'{col} RMSE' for col in df_pivot_rmse.columns]
df_pivot_rmse = df_pivot_rmse.reset_index()
print(df_pivot_rmse.to_csv(index=False, sep='\t'))

print('=' * 95)
\n
# --- END CELL ---
