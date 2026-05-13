import json

nb_path = r'C:\Users\LENOVO\Desktop\Physics Informed Neural Networks For EDA\Deep Ensemble\Deep_Ensemble_Inverse.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_cell_source = [
    '# 4. UQ BOUNDING\n',
    'threshold = 0.05\n',
    'success_mask = final_losses < threshold\n',
    'valid_widths = final_widths_phys[success_mask]\n',
    'valid_losses = final_losses[success_mask]\n',
    '\n',
    'if len(valid_widths) == 0:\n',
    '    print("No solutions found under strictly specified targets.")\n',
    'else:\n',
    '    K = min(5, len(valid_widths))\n',
    '    kmeans = KMeans(n_clusters=K, random_state=42).fit(valid_widths)\n',
    '\n',
    '    best_representatives = []\n',
    '    cluster_width_stds = []   # per-cluster width uncertainty\n',
    '\n',
    '    for cluster_id in range(K):\n',
    '        cluster_indices = np.where(kmeans.labels_ == cluster_id)[0]\n',
    '        cluster_widths = valid_widths[cluster_indices]  # all solutions in this cluster\n',
    '\n',
    '        # Best solution = lowest MSE loss inside this cluster\n',
    '        best_idx_in_cluster = cluster_indices[np.argmin(valid_losses[cluster_indices])]\n',
    '        best_representatives.append(best_idx_in_cluster)\n',
    '\n',
    '        # Width uncertainty = std of all valid solutions in this cluster.\n',
    '        # Interpretation: large +- means that width dimension has design flexibility;\n',
    '        # small +- means the optimizer tightly converged => that width is critical.\n',
    '        width_std = cluster_widths.std(axis=0)\n',
    '        cluster_width_stds.append(width_std)\n',
    '\n',
    '    print("\\n=======================================================")\n',
    '    print(" TOP DIVERSE GEOMETRIES")\n',
    '    print("=======================================================\\n")\n',
    '\n',
    '    for i, rep_idx in enumerate(best_representatives):\n',
    '        w_values = valid_widths[rep_idx]        # raw continuous widths from optimizer\n',
    '        width_uq = cluster_width_stds[i]        # +- per width for this design option\n',
    '\n',
    '        print(f"--- DESIGN OPTION #{i+1} ---")\n',
    '        for j, col in enumerate(width_names):\n',
    '            print(f"   > {col}: {w_values[j]:.3f}um  \u00b1 {width_uq[j]:.3f}um")\n',
    '\n',
    '        # Re-evaluate performance bounds using the raw optimizer widths\n',
    '        best_w_scaled = (torch.tensor(w_values, dtype=torch.float32, device=device) - mu_widths_1d) / std_widths_1d\n',
    '        best_full_scaled = torch.zeros((1, len(FEATURE_COLUMNS)), dtype=torch.float32, device=device)\n',
    '        best_full_scaled[0, width_indices] = best_w_scaled\n',
    '        best_full_scaled[0, fixed_indices] = fixed_scaled_vals[0]\n',
    '\n',
    '        # Deep Ensemble Evaluation: deterministic forward pass on all 5 models\n',
    '        preds_unscaled = []\n',
    '        for model in ensemble_models:\n',
    '            with torch.no_grad():\n',
    '                pred_eval, _ = model(best_full_scaled)\n',
    '                preds_unscaled.append(scaler_y_reg.inverse_transform(pred_eval.cpu().numpy())[0])\n',
    '\n',
    '        preds_unscaled = np.array(preds_unscaled)\n',
    '        mean_perf = preds_unscaled.mean(axis=0)\n',
    '        std_perf = preds_unscaled.std(axis=0)\n',
    '\n',
    '        print("\\n   [ PERFORMANCE PREDICTION \u00b1 UNCERTAINTY ]")\n',
    '        for idx, col in enumerate(REGRESSION_TARGETS):\n',
    '            target_v = target_metrics[col]\n',
    '            mean_v = mean_perf[idx]\n',
    '            uq_v = std_perf[idx] * 2  # 2-sigma: 95% confidence bounds\n',
    '            print(f"   - {col:16s} : Target={target_v:8.2f} | Out= {mean_v:8.2f} \u00b1 {uq_v:<7.2f} (Epistemic: {uq_v:.2f})")\n',
    '        print("\\n" + "="*55 + "\\n")\n'
]

# Find and patch the cell
patched = False
for nb_idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and '# 4.' in ''.join(cell['source']):
        nb['cells'][nb_idx]['source'] = new_cell_source
        nb['cells'][nb_idx]['outputs'] = []
        nb['cells'][nb_idx]['execution_count'] = None
        print(f'Patched cell at notebook index {nb_idx}')
        patched = True
        break

if not patched:
    print('ERROR: Could not find the target cell!')
else:
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print('Notebook saved successfully.')
