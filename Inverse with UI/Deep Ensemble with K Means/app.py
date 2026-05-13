"""
Deep Ensemble Inverse Predictor — Streamlit UI
================================================
Generates diverse transistor width combinations for a target set of
analog amplifier performance metrics using Multi-Start Adam optimisation
guarded by an ensemble of 5 PINNs (Epistemic-UQ Guardrail) and
K-Means clustering for diverse design-region discovery.
"""

import os
import sys
import warnings
import time
import pathlib

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import joblib
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import plotly.express as px

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PATHS  (resolve relative to this file so the app works from any CWD)
# ─────────────────────────────────────────────────────────────────────────────
APP_DIR   = pathlib.Path(__file__).resolve().parent          # …/Deep Ensemble with K Means
ENSEMBLE_DIR = APP_DIR.parent.parent / "Deep Ensemble"       # …/Deep Ensemble
DATA_PATH    = APP_DIR.parent.parent / "Data" / "FINAL_4CLASSES.csv"

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Deep Ensemble Inverse Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0d1117 0%, #161b27 50%, #0d1117 100%);
    color: #e6edf3;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #161b27 0%, #1c2333 100%);
    border-right: 1px solid #30363d;
}

/* Hero banner */
.hero-banner {
    background: linear-gradient(135deg, #6e40c9 0%, #2563eb 50%, #0891b2 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(99,60,180,0.4);
}
.hero-banner h1 {
    font-size: 2rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 0.3rem 0;
}
.hero-banner p {
    color: rgba(255,255,255,0.82);
    font-size: 0.95rem;
    margin: 0;
}

/* Section headers */
.section-header {
    font-size: 1.05rem;
    font-weight: 600;
    color: #58a6ff;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin: 1.2rem 0 0.6rem 0;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid #21262d;
}

/* Metric cards */
.metric-card {
    background: #161b27;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.metric-card:hover {
    border-color: #58a6ff;
    box-shadow: 0 0 12px rgba(88,166,255,0.15);
}
.metric-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: #8b949e;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.metric-value {
    font-size: 1.4rem;
    font-weight: 700;
    color: #58a6ff;
}
.metric-range {
    font-size: 0.7rem;
    color: #6e7681;
}

/* Design option cards */
.design-card {
    background: linear-gradient(135deg, #161b27, #1c2333);
    border: 1px solid #30363d;
    border-radius: 14px;
    padding: 1.4rem 1.8rem;
    margin-bottom: 1rem;
    transition: all 0.2s ease;
}
.design-card:hover {
    border-color: #6e40c9;
    box-shadow: 0 4px 24px rgba(110,64,201,0.2);
}
.design-card-header {
    font-size: 1.1rem;
    font-weight: 700;
    color: #e6edf3;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.8rem;
}
.badge {
    background: linear-gradient(90deg, #6e40c9, #2563eb);
    color: white;
    border-radius: 20px;
    padding: 0.15rem 0.65rem;
    font-size: 0.75rem;
    font-weight: 600;
}
.width-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.35rem 0;
    border-bottom: 1px solid #21262d;
    font-size: 0.9rem;
}
.width-name  { color: #8b949e; font-weight: 500; }
.width-value { color: #e6edf3; font-weight: 600; }
.width-uq    { color: #ffa657; font-size: 0.82rem; }
.perf-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.28rem 0;
    border-bottom: 1px solid #21262d;
    font-size: 0.85rem;
}
.perf-name   { color: #8b949e; min-width: 145px; }
.perf-target { color: #3fb950; font-weight: 500; }
.perf-out    { color: #58a6ff; font-weight: 600; }
.perf-uq     { color: #ffa657; }

/* Log / progress box */
.log-box {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-family: 'Courier New', monospace;
    font-size: 0.78rem;
    color: #7ee787;
    max-height: 180px;
    overflow-y: auto;
    white-space: pre-wrap;
}

/* Status pills */
.pill {
    display: inline-block;
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-size: 0.75rem;
    font-weight: 600;
}
.pill-green  { background: #1f4a2e; color: #3fb950; }
.pill-orange { background: #4a3000; color: #ffa657; }
.pill-blue   { background: #0c2d6b; color: #58a6ff; }
.pill-red    { background: #4a1010; color: #f85149; }

/* Buttons */
div.stButton > button {
    background: linear-gradient(90deg, #6e40c9, #2563eb);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.5rem 1.2rem;
    transition: all 0.2s;
    box-shadow: 0 2px 12px rgba(110,64,201,0.3);
}
div.stButton > button:hover {
    box-shadow: 0 4px 20px rgba(110,64,201,0.5);
    transform: translateY(-1px);
}

/* Number inputs */
div[data-testid="stNumberInput"] label,
div[data-testid="stSlider"] label {
    color: #8b949e;
    font-size: 0.85rem;
    font-weight: 500;
}

.stProgress > div > div {
    background: linear-gradient(90deg, #6e40c9, #2563eb);
}

/* Info boxes */
.info-box {
    background: #0c2d6b22;
    border-left: 3px solid #58a6ff;
    border-radius: 0 8px 8px 0;
    padding: 0.7rem 1rem;
    font-size: 0.85rem;
    color: #adbac7;
    margin: 0.6rem 0;
}

/* Expand button row */
.expand-row {
    text-align: center;
    margin: 0.5rem 0;
}

hr.divider {
    border: none;
    border-top: 1px solid #21262d;
    margin: 1.2rem 0;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLUMNS = [
    'Temperature(°)', 'W12(um)', 'W34(um)', 'W58(um)', 'W6(um)', 'W7(um)',
    'Idc(uA)', 'Length(um)', 'CC(pF)', 'CL(pF)'
]
REGRESSION_TARGETS = [
    'Gain(dB)', 'Bandwidth(Hz)', 'GBW(MHz)', 'Power(uW)', 'PM(degree)',
    'GM(dB)', 'PSRR(dB)', 'SlewRate (V/us)', 'CMRR(dB)'
]
WIDTH_NAMES = ['W12(um)', 'W34(um)', 'W58(um)', 'W6(um)', 'W7(um)']

# Dataset realistic ranges (from FINAL_4CLASSES.csv stats)
TARGET_RANGES = {
    'Gain(dB)':         (23.60, 54.23),
    'Bandwidth(Hz)':    (18219.10, 202933.00),
    'GBW(MHz)':         (3.29, 74.72),
    'Power(uW)':        (1659.71, 3422.44),
    'PM(degree)':       (44.60, 77.32),
    'GM(dB)':           (4.08, 14.36),
    'PSRR(dB)':         (32.60, 54.14),
    'SlewRate (V/us)':  (5.04, 9.92),
    'CMRR(dB)':         (54.80, 61.93),
}

TARGET_UNITS = {
    'Gain(dB)':         'dB',
    'Bandwidth(Hz)':    'Hz',
    'GBW(MHz)':         'MHz',
    'Power(uW)':        'µW',
    'PM(degree)':       '°',
    'GM(dB)':           'dB',
    'PSRR(dB)':         'dB',
    'SlewRate (V/us)':  'V/µs',
    'CMRR(dB)':         'dB',
}

# Fixed physical constants in the dataset (from the notebook)
FIXED_IDC    = 130.0   # µA
FIXED_LENGTH = 0.18    # µm
FIXED_CL     = 10.0    # pF
FIXED_CC     = 55.0    # pF

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─────────────────────────────────────────────────────────────────────────────
# PINN MODEL DEFINITION  (must match training architecture exactly)
# ─────────────────────────────────────────────────────────────────────────────
class PINN(nn.Module):
    def __init__(self, input_dim, hidden_dims, n_reg_outputs, n_classes, dropout_rate):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev_dim, h), nn.ReLU(), nn.Dropout(dropout_rate)]
            prev_dim = h
        self.backbone          = nn.Sequential(*layers)
        self.regression_head   = nn.Linear(prev_dim, n_reg_outputs)
        self.classification_head = nn.Linear(prev_dim, n_classes)

    def forward(self, x):
        shared = self.backbone(x)
        return self.regression_head(shared), self.classification_head(shared)


# ─────────────────────────────────────────────────────────────────────────────
# CACHED RESOURCE LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading ensemble models & scalers…")
def load_resources():
    scaler_X    = joblib.load(ENSEMBLE_DIR / "scaler_X.pkl")
    scaler_y    = joblib.load(ENSEMBLE_DIR / "scaler_y_reg.pkl")
    le          = joblib.load(ENSEMBLE_DIR / "label_encoder.pkl")
    n_classes   = len(le.classes_)

    models = nn.ModuleList()
    for i in range(1, 6):
        m = PINN(
            input_dim    = len(FEATURE_COLUMNS),
            hidden_dims  = [128, 128, 128, 128],
            n_reg_outputs= len(REGRESSION_TARGETS),
            n_classes    = n_classes,
            dropout_rate = 0.047,
        ).to(DEVICE)
        m.load_state_dict(torch.load(
            ENSEMBLE_DIR / f"pinn_ens_{i}.pth", map_location=DEVICE
        ))
        m.eval()
        for p in m.parameters():
            p.requires_grad = False
        models.append(m)

    return scaler_X, scaler_y, le, models


@st.cache_data(show_spinner=False)
def load_dataset():
    df = pd.read_csv(DATA_PATH, encoding='utf-8', engine='python')
    col_map = {
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
    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CORE INVERSE OPTIMISATION  (matches notebook exactly)
# ─────────────────────────────────────────────────────────────────────────────
def run_inverse_optimisation(
    target_metrics: dict,
    operating_conditions: dict,
    scaler_X, scaler_y, ensemble_models,
    n_starts: int = 1000,
    n_epochs: int = 1500,
    lambda_uq: float = 2.0,
    noise_scale: float = 1.5,
    progress_cb=None,
) -> tuple:
    """
    Returns:
        final_losses        (N_STARTS,)  MSE per start after optimisation
        final_widths_phys   (N_STARTS, 5)  physical widths
        mu_widths_1d, std_widths_1d, fixed_scaled_vals  — for re-evaluation
    """
    width_indices = [FEATURE_COLUMNS.index(c) for c in WIDTH_NAMES]
    fixed_indices = [FEATURE_COLUMNS.index(c) for c in operating_conditions.keys()]

    mu_widths_1d  = torch.tensor(scaler_X.mean_[width_indices],  dtype=torch.float32, device=DEVICE)
    std_widths_1d = torch.tensor(scaler_X.scale_[width_indices], dtype=torch.float32, device=DEVICE)

    inv_widths_raw = (
        mu_widths_1d
        + torch.randn((n_starts, len(WIDTH_NAMES)), device=DEVICE) * noise_scale
    ).clone().detach().requires_grad_(True)

    # Pre-compute fixed (operating-condition) scaled values
    fake_dict              = {**operating_conditions, **{k: 0 for k in WIDTH_NAMES}}
    initial_features_array = np.array([[fake_dict[col] for col in FEATURE_COLUMNS]])
    initial_scaled         = scaler_X.transform(initial_features_array)
    fixed_scaled_vals = (
        torch.tensor(initial_scaled[0, fixed_indices], dtype=torch.float32, device=DEVICE)
        .unsqueeze(0).repeat(n_starts, 1)
    )

    # Target tensor
    target_array  = np.array([[target_metrics[col] for col in REGRESSION_TARGETS]])
    target_scaled = scaler_y.transform(target_array)
    target_tensor = torch.tensor(target_scaled, dtype=torch.float32, device=DEVICE)

    optimizer    = optim.Adam([inv_widths_raw], lr=0.1)
    log_entries  = []
    t0           = time.time()

    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()

        widths_phys  = F.softplus(inv_widths_raw)
        widths_scaled = (widths_phys - mu_widths_1d) / std_widths_1d

        full_scaled = torch.zeros((n_starts, len(FEATURE_COLUMNS)), dtype=torch.float32, device=DEVICE)
        full_scaled[:, width_indices] = widths_scaled
        full_scaled[:, fixed_indices] = fixed_scaled_vals

        # Deep Ensemble forward pass
        all_preds = []
        for model in ensemble_models:
            p_reg, _ = model(full_scaled)
            all_preds.append(p_reg)

        stacked   = torch.stack(all_preds)                           # [5, N, 9]
        mean_pred = stacked.mean(dim=0)
        var_pred  = stacked.var(dim=0, unbiased=False)

        loss_mse = F.mse_loss(
            mean_pred, target_tensor.expand_as(mean_pred), reduction='none'
        ).mean(dim=1)
        loss_uq  = var_pred.mean(dim=1)

        total_loss = (loss_mse + lambda_uq * loss_uq).mean()
        total_loss.backward()
        optimizer.step()

        if epoch % 300 == 0 or epoch == n_epochs:
            msg = (
                f"Epoch {epoch:04d}/{n_epochs} | "
                f"Loss: {total_loss.item():.6f} | "
                f"MSE: {loss_mse.mean().item():.6f} | "
                f"UQ Var: {loss_uq.mean().item():.6f}"
            )
            log_entries.append(msg)
            if progress_cb:
                progress_cb(epoch / n_epochs, msg)

    # ── Re-evaluate on the FINAL updated weights (optimizer.step already ran) ──
    # The loop's last `loss_mse` is stale (computed before the last step),
    # so we do one clean forward pass with no_grad to get accurate final losses.
    final_widths_phys = F.softplus(inv_widths_raw).detach().cpu().numpy()

    with torch.no_grad():
        widths_phys_final  = F.softplus(inv_widths_raw)
        widths_scaled_final = (widths_phys_final - mu_widths_1d) / std_widths_1d
        full_scaled_final   = torch.zeros((n_starts, len(FEATURE_COLUMNS)), dtype=torch.float32, device=DEVICE)
        full_scaled_final[:, width_indices] = widths_scaled_final
        full_scaled_final[:, fixed_indices] = fixed_scaled_vals

        all_preds_final = []
        for model in ensemble_models:
            p_reg, _ = model(full_scaled_final)
            all_preds_final.append(p_reg)

        stacked_final    = torch.stack(all_preds_final)
        mean_pred_final  = stacked_final.mean(dim=0)
        final_losses     = F.mse_loss(
            mean_pred_final, target_tensor.expand_as(mean_pred_final), reduction='none'
        ).mean(dim=1).cpu().numpy()

    return (
        final_losses, final_widths_phys,
        mu_widths_1d, std_widths_1d,
        fixed_scaled_vals, fixed_indices, width_indices,
        log_entries
    )


def cluster_and_evaluate(
    final_losses, final_widths_phys,
    mu_widths_1d, std_widths_1d,
    fixed_scaled_vals, fixed_indices, width_indices,
    scaler_y, ensemble_models,
    target_metrics: dict,
    threshold: float = 0.05,
    K: int = 5,
) -> dict:
    """
    Applies threshold → K-Means → representative selection → per-design evaluation.
    Returns a rich results dict.
    """
    success_mask  = final_losses < threshold
    valid_widths  = final_widths_phys[success_mask]
    valid_losses  = final_losses[success_mask]
    n_valid       = len(valid_widths)

    if n_valid == 0:
        return {"status": "no_solutions", "n_valid": 0, "threshold": threshold}

    actual_K = min(K, n_valid)
    kmeans   = KMeans(n_clusters=actual_K, random_state=42, n_init="auto").fit(valid_widths)

    designs = []
    for cluster_id in range(actual_K):
        cluster_idx      = np.where(kmeans.labels_ == cluster_id)[0]
        cluster_widths   = valid_widths[cluster_idx]
        best_local_idx   = np.argmin(valid_losses[cluster_idx])
        best_global_idx  = cluster_idx[best_local_idx]

        w_values  = valid_widths[best_global_idx]
        width_std = cluster_widths.std(axis=0)
        mse_loss  = valid_losses[best_global_idx]

        # Re-evaluate with the 5 ensemble models (deterministic, no grad)
        best_w_scaled = (
            torch.tensor(w_values, dtype=torch.float32, device=DEVICE) - mu_widths_1d
        ) / std_widths_1d
        best_full = torch.zeros((1, len(FEATURE_COLUMNS)), dtype=torch.float32, device=DEVICE)
        best_full[0, width_indices] = best_w_scaled
        best_full[0, fixed_indices] = fixed_scaled_vals[0]

        preds_raw = []
        for model in ensemble_models:
            with torch.no_grad():
                pred, _ = model(best_full)
                preds_raw.append(scaler_y.inverse_transform(pred.cpu().numpy())[0])

        preds_raw  = np.array(preds_raw)
        mean_perf  = preds_raw.mean(axis=0)
        std_perf   = preds_raw.std(axis=0)    # 1-sigma
        uq_2sigma  = std_perf * 2             # 95 % CI

        designs.append({
            "cluster_id":   cluster_id,
            "w_values":     w_values,             # physical widths  [5]
            "width_uq":     width_std,            # per-width std    [5]
            "mse_loss":     float(mse_loss),
            "n_in_cluster": int(len(cluster_idx)),
            "mean_perf":    mean_perf,            # [9]
            "uq_2sigma":    uq_2sigma,            # [9] — 2-sigma
        })

    # Sort by mse_loss ascending (best first)
    designs.sort(key=lambda d: d["mse_loss"])

    return {
        "status":    "ok",
        "n_valid":   n_valid,
        "threshold": threshold,
        "K":         actual_K,
        "designs":   designs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_dataset_for_sampling():
    """Load dataset and return only the target columns for random sampling."""
    df = pd.read_csv(DATA_PATH, encoding='utf-8', engine='python')
    col_map = {
        'Gain(db)': 'Gain(dB)', 'Gain': 'Gain(dB)', 'gain': 'Gain(dB)',
        'Bandwidth': 'Bandwidth(Hz)', 'bandwidth': 'Bandwidth(Hz)',
        'GBW': 'GBW(MHz)', 'gbw': 'GBW(MHz)',
        'Power': 'Power(uW)', 'power': 'Power(uW)',
        'PM': 'PM(degree)', 'PhaseMargin': 'PM(degree)',
        'GM': 'GM(dB)', 'PSRR': 'PSRR(dB)',
        'SlewRate': 'SlewRate (V/us)', 'SlewRate(V/µs)': 'SlewRate (V/us)',
        'CMRR': 'CMRR(dB)',
    }
    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)
    return df


def random_target_metrics() -> dict:
    """Sample a REAL row from the dataset — guarantees physical consistency."""
    df = get_dataset_for_sampling()
    # Only sample rows that have all target columns
    available = [c for c in REGRESSION_TARGETS if c in df.columns]
    row = df[available].dropna().sample(n=1).iloc[0]
    return {col: float(row[col]) for col in REGRESSION_TARGETS}


def render_design_card(design: dict, idx: int, target_metrics: dict):
    """Renders a full design option card using native Streamlit widgets."""
    w_values  = design["w_values"]
    width_uq  = design["width_uq"]
    mean_perf = design["mean_perf"]
    uq2       = design["uq_2sigma"]
    cluster_size = design["n_in_cluster"]
    mse          = design["mse_loss"]

    with st.container(border=True):
        # ── Header ──────────────────────────────────────────────────────────
        hcol1, hcol2 = st.columns([1, 4])
        with hcol1:
            st.markdown(
                f"<span style='background:linear-gradient(90deg,#6e40c9,#2563eb);"
                f"color:white;border-radius:20px;padding:0.3rem 0.9rem;"
                f"font-size:1rem;font-weight:700;'>#{idx + 1}</span>",
                unsafe_allow_html=True,
            )
        with hcol2:
            st.markdown(
                f"**Design Region** &nbsp; "
                f"<span style='color:#6e7681;font-size:0.85rem;'>"
                f"Cluster: {cluster_size} solutions &nbsp;|&nbsp; MSE: {mse:.5f}"
                f"</span>",
                unsafe_allow_html=True,
            )

        st.divider()

        # ── Transistor Widths ± Uncertainty ──────────────────────────────
        st.markdown("**🔩 Transistor Widths ± Uncertainty (σ)**")
        w_df = pd.DataFrame({
            "Width Parameter": WIDTH_NAMES,
            "Value (µm)":      [f"{w_values[j]:.4f}" for j in range(len(WIDTH_NAMES))],
            "± Uncertainty":   [f"± {width_uq[j]:.4f} µm" for j in range(len(WIDTH_NAMES))],
        })
        st.dataframe(
            w_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Width Parameter": st.column_config.TextColumn(width="medium"),
                "Value (µm)":      st.column_config.TextColumn("Value (µm)", width="medium"),
                "± Uncertainty":   st.column_config.TextColumn("± Uncertainty (σ)", width="medium"),
            },
        )

        st.divider()

        # ── Performance Prediction ± 2σ ──────────────────────────────────
        st.markdown("**📊 Performance Prediction — Target vs Predicted (95 % CI = ±2σ)**")
        rows = []
        for pidx, col in enumerate(REGRESSION_TARGETS):
            tgt  = target_metrics[col]
            mean = mean_perf[pidx]
            uq   = uq2[pidx]
            err  = abs(mean - tgt) / (abs(tgt) + 1e-8) * 100
            unit = TARGET_UNITS.get(col, "")
            rows.append({
                "Metric":           col,
                "Target":           f"{tgt:.3f} {unit}",
                "Predicted (µ)":    f"{mean:.3f} {unit}",
                "± 2σ (Epistemic)": f"± {uq:.4f}",
                "Error %":          f"{err:.1f}%",
            })
        perf_df = pd.DataFrame(rows)
        st.dataframe(
            perf_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Metric":           st.column_config.TextColumn(width="medium"),
                "Target":           st.column_config.TextColumn(width="medium"),
                "Predicted (µ)":    st.column_config.TextColumn(width="medium"),
                "± 2σ (Epistemic)": st.column_config.TextColumn(width="medium"),
                "Error %":          st.column_config.TextColumn(width="small"),
            },
        )


def make_radar_chart(designs, target_metrics):
    """Radar chart comparing predicted vs target metrics (normalised)."""
    categories = REGRESSION_TARGETS
    n_cats     = len(categories)

    fig = go.Figure()

    # Target spoke (normalised to [0,1])
    mins = np.array([TARGET_RANGES[c][0] for c in categories])
    maxs = np.array([TARGET_RANGES[c][1] for c in categories])

    target_vals = np.array([target_metrics[c] for c in categories])
    target_norm = (target_vals - mins) / (maxs - mins + 1e-9)
    target_norm_closed = np.append(target_norm, target_norm[0])
    cats_closed        = categories + [categories[0]]

    fig.add_trace(go.Scatterpolar(
        r=target_norm_closed, theta=cats_closed,
        fill='toself',
        name='Target',
        line=dict(color='#3fb950', width=2),
        fillcolor='rgba(63,185,80,0.15)',
    ))

    colours = ['#58a6ff', '#ffa657', '#f85149', '#d2a8ff', '#79c0ff']
    for i, design in enumerate(designs):
        pred_norm = (design["mean_perf"] - mins) / (maxs - mins + 1e-9)
        pred_norm_closed = np.append(pred_norm, pred_norm[0])
        fig.add_trace(go.Scatterpolar(
            r=pred_norm_closed, theta=cats_closed,
            fill='toself',
            name=f'Design #{i+1}',
            line=dict(color=colours[i % len(colours)], width=1.5),
            fillcolor=f'rgba({int(colours[i%len(colours)][1:3],16)},{int(colours[i%len(colours)][3:5],16)},{int(colours[i%len(colours)][5:7],16)},0.08)',
        ))

    fig.update_layout(
        polar=dict(
            bgcolor='#161b27',
            radialaxis=dict(visible=True, range=[0, 1], color='#6e7681', gridcolor='#21262d'),
            angularaxis=dict(color='#adbac7', gridcolor='#21262d'),
        ),
        paper_bgcolor='#0d1117',
        plot_bgcolor='#0d1117',
        font=dict(color='#adbac7', family='Inter'),
        legend=dict(bgcolor='#161b27', bordercolor='#30363d', borderwidth=1),
        margin=dict(l=60, r=60, t=40, b=40),
        height=420,
    )
    return fig


def make_width_bar(designs):
    """Grouped bar chart of transistor widths across design options."""
    colours = ['#58a6ff', '#ffa657', '#f85149', '#d2a8ff', '#79c0ff']
    fig = go.Figure()
    for i, design in enumerate(designs):
        w    = design["w_values"]
        wuq  = design["width_uq"]
        fig.add_trace(go.Bar(
            name=f'Design #{i+1}',
            x=WIDTH_NAMES,
            y=w,
            error_y=dict(type='data', array=wuq, visible=True, color=colours[i % len(colours)]),
            marker_color=colours[i % len(colours)],
            opacity=0.85,
        ))
    fig.update_layout(
        barmode='group',
        paper_bgcolor='#0d1117',
        plot_bgcolor='#0d1117',
        font=dict(color='#adbac7', family='Inter'),
        yaxis=dict(title='Width (µm)', gridcolor='#21262d', color='#8b949e'),
        xaxis=dict(gridcolor='#21262d', color='#8b949e'),
        legend=dict(bgcolor='#161b27', bordercolor='#30363d', borderwidth=1),
        margin=dict(l=50, r=30, t=40, b=50),
        height=380,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ── Load resources ────────────────────────────────────────────────────────
    try:
        scaler_X, scaler_y, le, ensemble_models = load_resources()
    except Exception as e:
        st.error(f"❌ Failed to load models / scalers: {e}")
        st.info(f"Expected files in: `{ENSEMBLE_DIR}`")
        st.stop()

    # ── Hero Banner ───────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero-banner">
        <h1>⚡ Deep Ensemble Inverse Predictor</h1>
        <p>
            Multi-Start Adam Optimisation + Epistemic UQ Guardrail (5 PINNs) + K-Means Design Region Clustering<br>
            <span style="font-size:0.82rem; opacity:0.7;">
                Resolves the Many-To-One mapping problem for analog circuit sizing
            </span>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Optimiser Settings")
        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        n_starts = st.number_input(
            "Multi-Start Samples (N_STARTS)",
            min_value=100, max_value=5000, value=1000, step=100,
            help="Number of random initial width guesses fed to the optimiser."
        )
        n_epochs = st.number_input(
            "Optimisation Epochs",
            min_value=300, max_value=5000, value=1500, step=100,
        )
        lambda_uq = st.slider(
            "UQ Penalty Weight (λ)",
            min_value=0.0, max_value=10.0, value=2.0, step=0.5,
            help="Higher λ = stricter Epistemic UQ Guardrail."
        )
        threshold = st.slider(
            "MSE Acceptance Threshold",
            min_value=0.001, max_value=0.20, value=0.05, step=0.005,
            format="%.3f",
            help="Solutions with MSE < threshold are considered valid."
        )

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown("### 🔬 Design Regions")
        n_designs = st.number_input(
            "Design Regions to Show (K)",
            min_value=1, max_value=20, value=5,
            help="Number of K-Means clusters = diverse design solutions to display."
        )

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="font-size:0.78rem; color:#6e7681;">
            🖥️ Device: <code style="color:#58a6ff;">{DEVICE}</code><br>
            📦 Ensemble: <code style="color:#58a6ff;">5 PINNs</code><br>
            🏗️ Architecture: <code style="color:#58a6ff;">[128×128×128×128]</code><br>
            🎛️ Dropout: <code style="color:#58a6ff;">0.047</code>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Main two-column layout ────────────────────────────────────────────────
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        # ── Target Performance Metrics ────────────────────────────────────────
        st.markdown('<div class="section-header">🎯 Target Performance Metrics</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="info-box">Enter the desired analog amplifier specifications. '
            'Hit <b>🎲 Randomise</b> to fill in a realistic sample from the dataset range.</div>',
            unsafe_allow_html=True,
        )

        # Initialise session state for targets
        if "targets" not in st.session_state:
            st.session_state.targets = {}
            for t, (lo, hi) in TARGET_RANGES.items():
                st.session_state.targets[t] = float(np.mean([lo, hi]))

        if st.button("🎲 Randomise Targets", key="rand_targets"):
            rnd = random_target_metrics()
            for k, v in rnd.items():
                st.session_state.targets[k] = v
            st.rerun()

        user_targets = {}
        for metric in REGRESSION_TARGETS:
            lo, hi = TARGET_RANGES[metric]
            unit   = TARGET_UNITS.get(metric, "")
            val    = st.session_state.targets.get(metric, float(np.mean([lo, hi])))
            new_val = st.number_input(
                f"{metric} ({unit}) — range [{lo:.2f}, {hi:.2f}]",
                min_value=float(lo * 0.5),
                max_value=float(hi * 2.0),
                value=float(val),
                format="%.4f",
                key=f"tgt_{metric}",
            )
            user_targets[metric] = new_val
            # keep session state in sync
            st.session_state.targets[metric] = new_val

    with col_right:
        # ── Operating Conditions ──────────────────────────────────────────────
        st.markdown('<div class="section-header">🌡️ Operating Conditions</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="info-box">Physical operating conditions fed to the PINN as fixed inputs. '
            'Idc, Length, CC, CL are dataset constants; Temperature is the sweep variable.</div>',
            unsafe_allow_html=True,
        )

        temperature = st.number_input(
            "Temperature (°C) — range [-40, 125]",
            min_value=-40.0, max_value=125.0, value=27.0, step=1.0,
            key="temp_input",
        )
        st.number_input("Idc (µA) — fixed dataset constant",  value=FIXED_IDC,    disabled=True, key="idc")
        st.number_input("Length (µm) — fixed dataset constant", value=FIXED_LENGTH, disabled=True, key="length")
        st.number_input("CC (pF) — fixed dataset constant",     value=FIXED_CC,     disabled=True, key="cc")
        st.number_input("CL (pF) — fixed dataset constant",     value=FIXED_CL,     disabled=True, key="cl")

        operating_conditions = {
            'Temperature(°)': temperature,
            'Idc(uA)':        FIXED_IDC,
            'Length(um)':     FIXED_LENGTH,
            'CC(pF)':         FIXED_CC,
            'CL(pF)':         FIXED_CL,
        }

    # ── Generate Button ───────────────────────────────────────────────────────
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    gen_col, info_col = st.columns([1, 3])
    with gen_col:
        run_btn = st.button("🚀 Generate Design Regions", key="run_btn", use_container_width=True)

    # ── Results ───────────────────────────────────────────────────────────────
    if run_btn:
        st.session_state.pop("results", None)

        with st.status("Running Deep Ensemble Inverse Optimisation…", expanded=True) as status_box:
            progress_bar = st.progress(0.0, text="Initialising…")
            log_placeholder = st.empty()
            log_lines = []

            def progress_cb(frac, msg):
                progress_bar.progress(frac, text=msg)
                log_lines.append(msg)
                log_placeholder.markdown(
                    "<div class='log-box'>" + "\n".join(log_lines[-8:]) + "</div>",
                    unsafe_allow_html=True,
                )

            t_start = time.time()

            (
                final_losses, final_widths_phys,
                mu_widths_1d, std_widths_1d,
                fixed_scaled_vals, fixed_indices, width_indices,
                log_entries,
            ) = run_inverse_optimisation(
                target_metrics      = user_targets,
                operating_conditions= operating_conditions,
                scaler_X            = scaler_X,
                scaler_y            = scaler_y,
                ensemble_models     = ensemble_models,
                n_starts            = int(n_starts),
                n_epochs            = int(n_epochs),
                lambda_uq           = lambda_uq,
                progress_cb         = progress_cb,
            )

            elapsed = time.time() - t_start
            progress_bar.progress(1.0, text="Optimisation complete!")
            status_box.update(label=f"✅ Optimisation done in {elapsed:.1f}s", state="complete")

        # Cluster & evaluate
        results = cluster_and_evaluate(
            final_losses, final_widths_phys,
            mu_widths_1d, std_widths_1d,
            fixed_scaled_vals, fixed_indices, width_indices,
            scaler_y, ensemble_models,
            user_targets,
            threshold=threshold,
            K=int(n_designs),
        )
        results["target_metrics"]      = user_targets
        results["operating_conditions"]= operating_conditions
        st.session_state.results       = results

    # ── Display results if they exist in session state ────────────────────────
    if "results" in st.session_state:
        results       = st.session_state.results
        tgt_metrics   = results["target_metrics"]
        n_valid       = results["n_valid"]
        thr           = results["threshold"]

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        if results["status"] == "no_solutions":
            st.error(
                f"❌ No valid design regions found. "
                f"{n_valid} solutions crossed the MSE < {thr:.3f} threshold. "
                "Try relaxing the threshold or using a different target specification."
            )
        else:
            designs = results["designs"]
            K_actual = results["K"]

            # ── Summary stats ──────────────────────────────────────────────
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Valid Solutions", f"{n_valid:,}")
            s2.metric("Design Regions", K_actual)
            s3.metric("Acceptance Threshold (MSE)", f"{thr:.3f}")
            s4.metric("Best Region MSE", f"{designs[0]['mse_loss']:.5f}")

            # ── Charts ────────────────────────────────────────────────────
            st.markdown('<div class="section-header">📊 Design Region Analysis</div>', unsafe_allow_html=True)
            ch1, ch2 = st.columns(2)
            with ch1:
                st.plotly_chart(make_radar_chart(designs, tgt_metrics), use_container_width=True)
            with ch2:
                st.plotly_chart(make_width_bar(designs), use_container_width=True)

            # ── Design Cards ─────────────────────────────────────────────
            st.markdown(
                '<div class="section-header">🏗️ Design Regions (Best Representatives per Cluster)</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="info-box">'
                f'Showing top <b>{K_actual}</b> diverse design regions. '
                f'Each region is the <b>best-MSE</b> solution inside its K-Means cluster. '
                f'Width <b>± σ</b> = spread of all valid solutions in that cluster '
                f'(large ± → design flexibility; small ± → critical dimension). '
                f'Performance <b>± 2σ</b> = 95 % epistemic confidence from the 5-model ensemble.'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Default show top 5, allow expansion
            show_all_key   = "show_all_designs"
            if show_all_key not in st.session_state:
                st.session_state[show_all_key] = False

            N_DEFAULT = min(5, K_actual)
            display_n = K_actual if st.session_state[show_all_key] else N_DEFAULT

            for i, design in enumerate(designs[:display_n]):
                render_design_card(design, i, tgt_metrics)

            if K_actual > N_DEFAULT:
                remaining = K_actual - N_DEFAULT
                if not st.session_state[show_all_key]:
                    if st.button(
                        f"🔽 Show {remaining} more design region(s)",
                        key="show_more_btn"
                    ):
                        st.session_state[show_all_key] = True
                        st.rerun()
                else:
                    if st.button("🔼 Collapse to top 5", key="show_less_btn"):
                        st.session_state[show_all_key] = False
                        st.rerun()

            # ── Raw data download ──────────────────────────────────────────
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown('<div class="section-header">⬇️ Export Results</div>', unsafe_allow_html=True)

            rows = []
            for i, d in enumerate(designs):
                row = {"Design": f"#{i+1}", "MSE": d["mse_loss"], "Cluster_Size": d["n_in_cluster"]}
                for j, wn in enumerate(WIDTH_NAMES):
                    row[f"{wn}_mean"] = d["w_values"][j]
                    row[f"{wn}_uq"]   = d["width_uq"][j]
                for pidx, col in enumerate(REGRESSION_TARGETS):
                    row[f"{col}_target"]    = tgt_metrics[col]
                    row[f"{col}_predicted"] = d["mean_perf"][pidx]
                    row[f"{col}_uq2sigma"]  = d["uq_2sigma"][pidx]
                rows.append(row)

            result_df = pd.DataFrame(rows)
            st.download_button(
                "📥 Download Results as CSV",
                data=result_df.to_csv(index=False).encode(),
                file_name="deep_ensemble_inverse_results.csv",
                mime="text/csv",
            )
            st.dataframe(result_df, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
