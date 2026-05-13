import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(12, 6))

# Hide axes
ax.axis('off')

# Function to draw a box
def draw_box(ax, x, y, width, height, text, title=None, color='#e0f7fa'):
    # Box
    rect = patches.FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.1", 
                                  edgecolor='black', facecolor=color, lw=1.5)
    ax.add_patch(rect)
    # Text
    if title:
        ax.text(x + width/2, y + height - 0.15, title, ha='center', va='center', 
                fontsize=11, fontweight='bold', color='black')
        ax.text(x + width/2, y + height/2 - 0.1, text, ha='center', va='center', 
                fontsize=10, color='black')
    else:
        ax.text(x + width/2, y + height/2, text, ha='center', va='center', 
                fontsize=10, color='black', fontweight='bold')

# Draw Forward Process
draw_box(ax, 0.5, 3.5, 2.5, 1.2, "Fixed Geometric Layout\n(W1, W2, ...)", title="Inputs", color='#c8e6c9')
ax.annotate('', xy=(3.5, 4.1), xytext=(3.0, 4.1), arrowprops=dict(arrowstyle="->", lw=2))

draw_box(ax, 3.5, 3.5, 3.5, 1.2, "5 x PINN Models\n(Different Seeds)", title="Deep Ensemble", color='#fff9c4')
ax.annotate('', xy=(7.5, 4.1), xytext=(7.0, 4.1), arrowprops=dict(arrowstyle="->", lw=2))

draw_box(ax, 7.5, 3.5, 4.0, 1.2, "Gain: 50dB ± 1.5dB\n*True Model (Epistemic) Uncertainty*", title="Predicted Performance", color='#ffccbc')

# Draw Inverse Process
draw_box(ax, 0.5, 1.0, 2.5, 1.2, "Required Specs\n(Gain > 50dB, etc.)", title="Target Performance", color='#c8e6c9')
ax.annotate('', xy=(3.5, 1.6), xytext=(3.0, 1.6), arrowprops=dict(arrowstyle="->", lw=2))

draw_box(ax, 3.5, 1.0, 3.5, 1.2, "1000-Start Adam Optimization\n+ K-Means Clustering", title="Inverse Optimizer", color='#fff9c4')
ax.annotate('', xy=(7.5, 1.6), xytext=(7.0, 1.6), arrowprops=dict(arrowstyle="->", lw=2))

draw_box(ax, 7.5, 1.0, 4.0, 1.2, "W1: 10µm ± 0.5µm\n*Design Tolerance (Geometric Margin)*", title="Predicted Geometry", color='#ffccbc')

# Titles
ax.text(6, 5.2, "Forward UQ: Output Model Uncertainty", ha='center', fontsize=14, fontweight='bold', color='#1a237e')
ax.text(6, 2.7, "Inverse UQ: Geometric Design Flexibility", ha='center', fontsize=14, fontweight='bold', color='#1a237e')

# Draw dividing line
plt.axhline(y=2.8, color='black', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(r'c:\Users\LENOVO\Desktop\Physics Informed Neural Networks For EDA\uq_comparison_diagram.png', dpi=300, bbox_inches='tight')
plt.close()
