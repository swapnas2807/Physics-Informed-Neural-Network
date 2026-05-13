import tkinter as tk
from tkinter import ttk, messagebox
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import joblib
import numpy as np
import threading
import sys

# ======================================================================
# 1. Define Model Architecture (Must match your training script)
# ======================================================================
# These must match the notebook
HIDDEN_DIMS = [128, 128, 128]
FEATURE_COLUMNS = [
    'Temperature(°)', 'W12(um)', 'W34(um)', 'W58(um)', 'W6(um)', 'W7(um)', 
    'Idc(uA)', 'Length(um)', 'CC(pF)'
]
REGRESSION_TARGETS = [
    'Gain(dB)', 'Bandwidth(Hz)', 'GBW(MHz)', 'Power(uW)', 'PM(degree)', 
    'GM(dB)', 'PSRR(dB)', 'SlewRate (V/us)', 'CMRR(dB)'
]
N_CLASSES = 5 # From your notebook output
WIDTH_NAMES = ['W12(um)', 'W34(um)', 'W58(um)', 'W6(um)', 'W7(um)']

class PINN(nn.Module):
    def __init__(self, input_dim, hidden_dims, n_reg_outputs, n_classes):
        super(PINN, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.regression_head = nn.Linear(prev_dim, n_reg_outputs)
        self.classification_head = nn.Linear(prev_dim, n_classes)

    def forward(self, x):
        shared_output = self.backbone(x)
        reg_output = self.regression_head(shared_output)
        class_output = self.classification_head(shared_output)
        return reg_output, class_output

# ======================================================================
# 2. Load Models and Scalers (Global)
# ======================================================================
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load scalers
    scaler_X = joblib.load('scaler_X.pkl')
    scaler_y_reg = joblib.load('scaler_y_reg.pkl')
    
    # Load model architecture
    inverse_model = PINN(
        input_dim=len(FEATURE_COLUMNS),
        hidden_dims=HIDDEN_DIMS,
        n_reg_outputs=len(REGRESSION_TARGETS),
        n_classes=N_CLASSES
    ).to(device)
    
    # Load trained weights
    inverse_model.load_state_dict(torch.load("final_rm_optipinn_model.pth", map_location=device))
    inverse_model.eval()
    print(f"Model and scalers loaded successfully on {device}.")

except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    print("Please make sure 'final_rm_optipinn_model.pth', 'scaler_X.pkl', and 'scaler_y_reg.pkl' are in the same folder.")
    messagebox.showerror("File Not Found", f"Error: {e}\n\nPlease make sure all required files are in the script's folder.")
    sys.exit()

# ======================================================================
# 3. Core Inverse Prediction Logic (Option C from our chat)
# ======================================================================
def run_inverse_prediction(target_performance_dict, target_temp, fixed_consts):
    """
    Runs the inverse prediction in a separate thread.
    """
    with torch.no_grad():
        # 1. Scale Target Performance
        target_perf_values = np.array([target_performance_dict[m] for m in REGRESSION_TARGETS]).reshape(1, -1)
        target_perf_scaled = torch.tensor(scaler_y_reg.transform(target_perf_values), dtype=torch.float32).to(device)

        # 2. Prepare Fixed Inputs (Temp, Idc, L, CC)
        # -- Temperature (Index 0) --
        mu_temp = torch.tensor(scaler_X.mean_[0], dtype=torch.float32, device=device)
        std_temp = torch.tensor(scaler_X.scale_[0], dtype=torch.float32, device=device)
        val_temp_scaled = (torch.tensor(target_temp, device=device) - mu_temp) / std_temp
        val_temp_scaled = val_temp_scaled.reshape(1, 1)

        # -- Constants: Idc(6), L(7), CC(8) --
        # We manually scale them. Since std=0, we just set scaled to 0.
        consts_scaled = torch.zeros((1, 3), dtype=torch.float32, device=device)

        # 3. Initialize Trainable Widths (Indices 1-5)
        mu_widths = torch.tensor(scaler_X.mean_[1:6], dtype=torch.float32, device=device)
        std_widths = torch.tensor(scaler_X.scale_[1:6], dtype=torch.float32, device=device)
        
        # Use torch.no_grad() for initialization, then enable grad
        init_widths = mu_widths.unsqueeze(0)
        inv_widths_raw = init_widths.clone().detach().requires_grad_(True)
    
    # Optimizer targeting ONLY the widths
    inv_optimizer = optim.Adam([inv_widths_raw], lr=0.1)
    
    # 4. Optimization Loop
    num_steps = 1000
    for step in range(num_steps):
        inv_optimizer.zero_grad()
        
        widths_phys = F.softplus(inv_widths_raw)
        widths_scaled = (widths_phys - mu_widths) / std_widths
        
        full_input_scaled = torch.cat([val_temp_scaled, widths_scaled, consts_scaled], dim=1)
        
        pred_reg_scaled, _ = inverse_model(full_input_scaled)
        
        loss_inv = F.mse_loss(pred_reg_scaled, target_perf_scaled)
        loss_inv.backward()
        inv_optimizer.step()

    # 5. Decode Final Results
    with torch.no_grad():
        final_widths = F.softplus(inv_widths_raw).cpu().numpy()[0]
        
        inverse_result = {
            'W12(um)': final_widths[0],
            'W34(um)': final_widths[1],
            'W58(um)': final_widths[2],
            'W6(um)': final_widths[3],
            'W7(um)': final_widths[4],
        }
        
    return inverse_result

# ======================================================================
# 4. Tkinter GUI Application
# ======================================================================
class CircuitInverterApp:
    def __init__(self, root):
        self.root = root
        root.title("Circuit Inverse Design (RM-OptiPINN)")
        root.geometry("600x650")

        # Style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- 1. Target Performance Inputs ---
        target_frame = ttk.Labelframe(main_frame, text="Target Performance Metrics", padding="10")
        target_frame.pack(fill=tk.X, expand=True, pady=5)
        
        self.target_entries = {}
        # Pre-filled example values from your notebook
        default_targets = {
            'Gain(dB)': 52.2662, 'Bandwidth(Hz)': 25126.0, 'GBW(MHz)': 28.5279,
            'Power(uW)': 2145.86, 'PM(degree)': 64.1203, 'GM(dB)': 8.36547,
            'PSRR(dB)': 48.3298, 'SlewRate (V/us)': 6.8029, 'CMRR(dB)': 60.292
        }
        
        for i, name in enumerate(REGRESSION_TARGETS):
            row = i // 2
            col = (i % 2) * 2
            ttk.Label(target_frame, text=name).grid(row=row, column=col, padx=5, pady=5, sticky="w")
            var = tk.StringVar(value=f"{default_targets.get(name, 0.0)}")
            entry = ttk.Entry(target_frame, textvariable=var, width=15)
            entry.grid(row=row, column=col + 1, padx=5, pady=5)
            self.target_entries[name] = var

        # --- 2. Fixed Conditions Inputs ---
        cond_frame = ttk.Labelframe(main_frame, text="Fixed Operating Conditions", padding="10")
        cond_frame.pack(fill=tk.X, expand=True, pady=5)
        
        self.condition_entries = {}
        
        # Temperature (User input)
        ttk.Label(cond_frame, text="Temperature(°)").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        temp_var = tk.StringVar(value="-40.0") # Default to room temp
        ttk.Entry(cond_frame, textvariable=temp_var, width=15).grid(row=0, column=1, padx=5, pady=5)
        self.condition_entries['Temperature(°)'] = temp_var
        
        # Constants (Prefilled, read-only)
        constants = {'Idc(uA)': 130.0, 'Length(um)': 0.18, 'CC(pF)': 55.0}
        for i, (name, val) in enumerate(constants.items(), 1):
            ttk.Label(cond_frame, text=name).grid(row=i, column=0, padx=5, pady=5, sticky="w")
            var = tk.StringVar(value=f"{val}")
            entry = ttk.Entry(cond_frame, textvariable=var, width=15, state="readonly")
            entry.grid(row=i, column=1, padx=5, pady=5)
            self.condition_entries[name] = var # Not really needed, but good practice

        # --- 3. Prediction Button ---
        self.predict_button = ttk.Button(main_frame, text="Predict Widths", command=self.start_prediction_thread)
        self.predict_button.pack(pady=10, fill=tk.X)
        
        # --- 4. Results Display ---
        result_frame = ttk.Labelframe(main_frame, text="Predicted Widths", padding="10")
        result_frame.pack(fill=tk.X, expand=True, pady=5)
        
        self.result_vars = {}
        for i, name in enumerate(WIDTH_NAMES):
            ttk.Label(result_frame, text=name).grid(row=i, column=0, padx=5, pady=5, sticky="w")
            var = tk.StringVar(value="---")
            entry = ttk.Entry(result_frame, textvariable=var, width=20, state="readonly", justify="center")
            entry.grid(row=i, column=1, padx=5, pady=5)
            self.result_vars[name] = var
            
        # --- 5. Status Bar ---
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor="w", padding="2 5")
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def start_prediction_thread(self):
        """
        Starts the inverse prediction in a separate thread to avoid freezing the GUI.
        """
        self.predict_button.config(state="disabled")
        self.status_var.set("Working... (This may take 10-20 seconds)")
        
        # Launch in a daemon thread
        threading.Thread(target=self.run_prediction, daemon=True).start()

    def run_prediction(self):
        """
        Gathers inputs, runs model, and posts results back to GUI.
        (This runs in the background thread)
        """
        try:
            # 1. Gather Target Inputs
            target_dict = {}
            for name, var in self.target_entries.items():
                target_dict[name] = float(var.get())
            
            # 2. Gather Condition Inputs
            target_temp = float(self.condition_entries['Temperature(°)'].get())
            fixed_consts = {
                'Idc(uA)': float(self.condition_entries['Idc(uA)'].get()),
                'Length(um)': float(self.condition_entries['Length(um)'].get()),
                'CC(pF)': float(self.condition_entries['CC(pF)'].get())
            }

            # 3. Run Model
            result_dict = run_inverse_prediction(target_dict, target_temp, fixed_consts)
            
            # 4. Post results back to main thread
            self.root.after(0, self.update_gui_results, result_dict, None)

        except ValueError as e:
            # Handle bad user input (e.g., "abc" in an entry box)
            self.root.after(0, self.update_gui_results, None, f"Input Error: Please enter valid numbers. {e}")
        except Exception as e:
            # Handle any other model errors
            self.root.after(0, self.update_gui_results, None, f"Error: {e}")

    def update_gui_results(self, result_dict, error_msg):
        """
        Updates the GUI with results or an error message.
        (This runs safely in the main thread)
        """
        if error_msg:
            self.status_var.set(f"Error: {error_msg}")
            messagebox.showerror("Error", error_msg)
        elif result_dict:
            for name, var in self.result_vars.items():
                var.set(f"{result_dict[name]:.6f} um")
            self.status_var.set("Done!")
        
        self.predict_button.config(state="normal")


# ======================================================================
# 5. Run the Application
# ======================================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = CircuitInverterApp(root)
    root.mainloop()