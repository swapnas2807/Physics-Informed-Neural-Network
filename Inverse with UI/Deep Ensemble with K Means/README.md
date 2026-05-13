# Deep Ensemble Inverse Predictor — Streamlit UI

## How to Run

```bash
cd "Inverse with UI/Deep Ensemble with K Means"
streamlit run app.py
```

The app will open at **http://localhost:8501**

## What it does

1. **Input Target Metrics** — Enter the 9 analog amplifier performance specs you want to achieve.  
   Hit **🎲 Randomise Targets** to fill with a realistic random sample from the dataset range.

2. **Operating Conditions** — Temperature is adjustable. Idc, Length, CC, CL are locked to the dataset constants used during training.

3. **Sidebar Controls:**
   - `N_STARTS` — number of random initial guess geometries (default 1000)
   - `Epochs` — Adam optimisation steps (default 1500)
   - `λ (UQ Penalty)` — weight of the Epistemic UQ guardrail (default 2.0)
   - `MSE Threshold` — acceptance cutoff; lower = stricter (default 0.05)
   - `K (Design Regions)` — how many diverse clusters to return (default 5)

4. **🚀 Generate** — Runs multi-start Adam optimisation over 1000 starts with the deep ensemble guardrail, clusters valid results with K-Means, picks the best representative per cluster.

5. **Results:**
   - **Radar chart** — normalised target vs predicted per design region
   - **Width bar chart** — W12/W34/W58/W6/W7 comparison with ±σ error bars
   - **Design cards** — widths ± cluster spread, performance ± 2σ epistemic confidence
   - **Show more** button to view additional design regions beyond the top 5
   - **CSV export** of all results

## Dependencies

```
streamlit
torch
numpy
pandas
joblib
scikit-learn
plotly
```
