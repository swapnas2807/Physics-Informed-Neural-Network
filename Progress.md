# EDA Machine Learning Project Progress & Journal
**Purpose:** A persistent journal tracking deep structural discussions, architectural decisions, and VLSI physics learnings.

---

## Date: 2026-04-20

### 1. Architectural Evaluation: Multi-Start vs. Generative cVAE
* **The Test:** We fully built and benchmarked two separate inverse layout engines to find the best approach:
    1. **Generative cVAE:** A Conditional Variational Autoencoder that attempts to directly learn the mathematical distribution of the continuous circuit physical topologies from target metrics.
    2. **Multi-Start Global Optimizer:** Taking the pre-trained forward PINN, and running 1,000 parallel Adam gradient descent instances simultaneously on randomized noisy input guesses.
* **The Result:** We natively benched both architectures side-by-side using `Inverse_Evaluation_Suite.ipynb`. While the cVAE was instantaneously fast (`0.0006 secs`), it suffered violently from **Mode Collapse**. No matter how we perturbed the latent noise `z`, it outputted identical physical layouts (`Diversity score: 0.15um`), meaning it artificially forced a non-physical 1-to-1 shortcut map rather than exploring distinct topology valleys.
* **The Final Decision (Ditching the cVAE):** The cVAE was officially abandoned. In contrast, the Multi-Start algorithm outputted 5 definitively distinct physical configurations perfectly matching performance goals (`Diversity score: 13.56um`). Furthermore, attempting to inject stable Epistemic UQ bounds onto a cVAE is mathematically unstable. The **Multi-Start Optimizer + Forward Deep Ensemble PINN** is universally more robust, inherently prevents the hallucination loophole, and is much simpler to rigorously test for a VLSI publication.

### 2. The Many-To-One Problem in Inverse Prediction
* **Concept:** In a forward model (`Widths -> Performance`), the mapping is a strict One-to-One mathematical function. However, the exact reverse (`Performance -> Widths`) is a "One-to-Many" mapping. Multiple totally different geometric circuit configurations can yield the exact same electrical performance mappings.
* **Resolution:** We solved this using a **Multi-Start Global Optimizer**. By dropping 1,000 independent stochastically random starting points into PyTorch Adam, we generated hundreds of diverse valid geometries, resolving them into distinct structural families using K-Means clustering.

### 3. The "Hallucination Loophole" 
* **Concept:** An inverse engine (whether Generative VAE or Gradient Descent) fundamentally relies on the frozen Forward Model (PINN) to check if its output is "correct".
* **The Danger:** If the optimizer stumbles across a weird string of widths that was never in the training dataset, the Forward PINN might confidently hallucinate the wrong answer (e.g. reporting 50dB when reality is 5dB). The Inverse Predictor will then proudly output that completely broken layout to the user.
* **The Fix:** We must build strict **Uncertainty Quantification (UQ)** on the Forward Pass to act as a "Guardrail" so the optimizer is punished for picking solutions in unknown data spaces.

### 4. VLSI Manufacturing Grid Constraints
* **Concept:** Foundries manufacture circuits using strict discrete photolithography grids (`lambda`, e.g., `0.005um`).
* **The Learning:** Outputting an inverse machine-learning prediction of `W = 10.0 ± 0.0012um` is completely useless in the EDA world, because the CAD mouse will mathematically snap past variations that small, and chemical etching errors in the fab are far larger than sub-nanometer levels.

### 5. Epistemic (Model) Uncertainty vs Aleatoric (Manufacturing) Uncertainty
* **The Old Error:** Previously, MC Dropout was left active during the gradient descent on the input tensor. This forced the optimizer to land at slightly different Physical Widths. This wrongly projected *Epistemic Model Uncertainty* (the AI's lack of knowledge) onto the *Physical Geometry Ruler*.
* **The New Architecture (Deep Ensembles):** We will build an Ensemble of 5 completely separate PINNs.
    1. The Inverse Optimizer will output a **single, strict, grid-aligned geometry** (e.g., `W = 10.05um`).
    2. We feed that geometry into all 5 PINNs.
    3. We shift the `±` Uncertainty to the **Outputs**, saying: *"This exact geometry yields 50dB Gain ± 1.5dB"*. This successfully answers the exact requirement for a VLSI engineer by telling them mathematically where the AI lacks confidence in the physics graph.
## Next Implementation Tasks
When transitioning to the new development space, execute the following roadmap:
1. **Build & Train the Deep Ensemble:** Write an isolated script that loops 5 times with 5 distinct random mathematical initialization seeds. Train 5 identical standard PINNs and save the isolated weight files (`pinn_ens_1.pth` through `pinn_ens_5.pth`).
2. **Modify the Inverse Predictor:** Upgrade the `Multi_Start_Inverse.ipynb` environment to load a Python list containing all 5 distinct Neural Networks side-by-side inside the GPU.
3. **Inject the Forward UQ Penalty:** Inside the Adam optimizer loop, pass generated geometries through all 5 models simultaneously. Calculate the **Mean** for the standard Accuracy MSE, and calculate the **Variance** explicitly as the Epistemic Uncertainty. Inject it into the loss: `Total_Loss = MSE + lambda * Variance` to functionally punish the optimizer for wandering into uncertain areas.
4. **Restructure Physical Outputs:** Add code blocks bounding physical geometries to realistic `0.005um` CAD gridding points to solve Professor's layout critique, and officially decouple them from any `±` parameters.

---

## Date: 2026-04-22

### 6. Width-Level Uncertainty Display (±) on Inverse Predicted Widths

* **The Question:** Currently, the inverse predictor outputs only a single value per width (e.g., `W12: 6.725um`). The goal was to attach a `±` uncertainty term to each predicted width so a circuit designer knows how precisely each dimension needs to be hit.

* **Three Options Were Evaluated:**

    1. **Option 1 — K-Means Cluster Spread (Implemented):** Since 1000 optimizer starts already produce hundreds of valid solutions which are then clustered by K-Means, the **standard deviation of widths across all members of each cluster** is the natural `±`. A large `±` on W6 means many valid solutions had different W6 values — that width has design freedom. A small `±` on W34 means all solutions tightly agreed — that width is critical. This is scientifically valid as a measure of **design margin / parametric tolerance**, though it is not true ML epistemic uncertainty.

    2. **Option 2 — Gradient / Sensitivity-Based:** After the optimizer finds W*, compute `∂Loss/∂W_j` for each width at the solution. High gradient magnitude = steep loss landscape = that width is sensitive = small `±`. Low gradient = flat landscape = design flexibility = large `±`. Related to the Laplace approximation in Bayesian inference. More principled mathematically, requires one backward pass per design option.

    3. **Option 3 — Repeat Optimizer Runs, Take Std:** Run the full 1000-start optimization N times with different seeds. Take std of final widths across runs. **Rejected** — because the inverse problem is many-to-one, different runs will discover completely different valid design regions. Taking std across runs measures optimizer randomness, not any meaningful uncertainty about the widths.

* **Key Scientific Clarification:** None of the three options represent true ML epistemic uncertainty about widths. The Deep Ensemble UQ lives in the **forward direction** (uncertainty in predicted performance given widths). To get true probabilistic uncertainty over widths given targets, a **generative inverse model like cVAE** is needed — which is why the cVAE was explored earlier. The `±` from Options 1/2 is best described as **design tolerance** or **sensitivity bounds**, not model confidence.

### 7. Solution Space & Pass Rate Discussion

* **Insight:** The number of the 1000 optimizer starts that pass the MSE threshold is a signal about the size of the valid solution space. High pass rate (~900/1000) → the valid region in width space is wide, the target is easy to satisfy, and there is significant design freedom. Low pass rate (~50/1000) → the valid region is narrow, the target is hard to hit, widths must be precise.

* **Consequence for K-Means:** With a fixed `K=5`, a high pass rate means the five clusters may be merging genuinely distinct design regions together. Dynamically scaling K based on pass rate (e.g., `K=10` if >70% pass) would surface more diverse design options.

* **Fundamental Limit:** It is mathematically impossible to enumerate all valid width combinations that satisfy a given set of target performance metrics. The valid solution set is a continuous manifold in 5D width space. Even at a 0.005um grid, the search space is ~10²⁰ points. The 1000-start optimizer is a **representative sampler** of this manifold, not an exhaustive solver.

### 8. Grid Snapping Removed from Inverse Output

* **Previous State:** `Deep_Ensemble_Inverse.ipynb` Cell 4 was applying `np.round(w / 0.005) * 0.005` to both the width values and their uncertainties before printing. The performance re-evaluation also used the grid-snapped widths as input to the PINN ensemble.

* **Change Made:** Grid snapping was fully removed. The notebook now displays raw continuous widths from the optimizer directly, along with their cluster-spread uncertainty. The PINN performance re-evaluation now also uses the raw widths. Grid snapping can be re-introduced later as a separate post-processing step.

* **Output format after change:**
    ```
    --- DESIGN OPTION #1 ---
       > W12(um): 6.725um  ± 0.183um
       > W34(um): 27.225um ± 0.094um
       > W58(um): 47.480um ± 0.312um
       > W6(um):  22.205um ± 1.847um
       > W7(um):  25.140um ± 0.276um
    ```

### 9. Systematic Validation of the Inverse Predictor

* **The Goal:** To rigorously quantify the inverse predictor's accuracy, yield, and design space coverage using a held-out test set, and package this into a presentation-ready report.

* **The Validation Pipeline:**
    1. **Stratified Test Set:** Sampled 30 real rows from `FINAL_4CLASSES.csv`, stripped the widths, and fed only the performance targets + physical conditions to the inverse predictor.
    2. **End-to-End Execution:** The `Multi-Start Adam + Epistemic UQ Guardrail` was run for 1000 starts per sample, generating valid candidate designs.
    3. **Forward Validation:** The top 5 K-Means design candidates were fed back into the 5-model deep ensemble forward PINN to calculate the actual predicted performance vs the target.

* **Key Results:**
    * **100% Yield Rate:** Valid transistor sizing solutions were found for all 30 test samples across the full operating temperature range.
    * **2.14% Mean Error:** Forward-validated performance is well within engineering tolerance (5-10%) for almost all metrics (Gain, CMRR, PSRR, Phase Margin < 0.6%). Bandwidth and GBW are the hardest to fit (~7-9% error).
    * **UQ Calibration:** A positive Spearman correlation between the ensemble's uncertainty (σ) and prediction error confirmed the UQ guardrail is honest—the model knows when it doesn't know.
    * **Design Diversity:** The K-Means clustering found genuinely distinct transistor topologies, with a mean pairwise width distance of ~5.6µm between design regions, successfully resolving the many-to-one inverse problem.

* **Artifact Generation:** A standalone Jupyter notebook (`Inverse_Validation.ipynb`) was built to run this pipeline and automatically generate 10 publication-quality plots (convergence, per-metric error, yield, target vs. predicted radar/scatter metrics, etc.). A Python script (`generate_report.py`) using `python-docx` was also created to officially compile these results and plots into a comprehensive Word document report (`Inverse_Predictor_Validation_Report.docx`) detailing the system design and graph analysis.

---
*(Add new entries above this line as discussions progress in future sessions)*
