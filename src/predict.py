import torch
import numpy as np
import matplotlib.pyplot as plt
from model import PINN

# ------------------------------------------------------------
# Settings (must match train.py)
# ------------------------------------------------------------
alpha = 1.0

# ------------------------------------------------------------
# Load the trained model
# ------------------------------------------------------------
model = PINN(hidden_layers=4, neurons=20)
model.load_state_dict(torch.load("trained_model.pth"))
model.eval()   # switch to evaluation mode (no training)

# ------------------------------------------------------------
# Create a grid of (x, t) points to evaluate the solution
# ------------------------------------------------------------
n_points = 100
x_vals = np.linspace(0, 1, n_points)
t_vals = np.linspace(0, 1, n_points)

X, T = np.meshgrid(x_vals, t_vals)   # 2D grid

# Flatten the grid and convert to torch tensors
x_flat = torch.tensor(X.flatten(), dtype=torch.float32).view(-1, 1)
t_flat = torch.tensor(T.flatten(), dtype=torch.float32).view(-1, 1)

# ------------------------------------------------------------
# Get PINN prediction (no gradients needed here)
# ------------------------------------------------------------
with torch.no_grad():
    u_pred = model(x_flat, t_flat).numpy().reshape(n_points, n_points)

# ------------------------------------------------------------
# Compute the exact analytical solution for comparison
# u(x,t) = sin(pi*x) * exp(-pi^2 * alpha * t)
# ------------------------------------------------------------
u_exact = np.sin(np.pi * X) * np.exp(-(np.pi**2) * alpha * T)

# ------------------------------------------------------------
# Plot: PINN prediction vs Exact solution
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

im1 = axes[0].pcolormesh(X, T, u_pred, shading='auto', cmap='hot')
axes[0].set_title("PINN Prediction")
axes[0].set_xlabel("x")
axes[0].set_ylabel("t")
fig.colorbar(im1, ax=axes[0])

im2 = axes[1].pcolormesh(X, T, u_exact, shading='auto', cmap='hot')
axes[1].set_title("Exact Solution")
axes[1].set_xlabel("x")
axes[1].set_ylabel("t")
fig.colorbar(im2, ax=axes[1])

plt.tight_layout()

# ------------------------------------------------------------
# Save the figure into the results folder
# ------------------------------------------------------------
plt.savefig("../results/heat_solution.png", dpi=150)
print("Plot saved to results/heat_solution.png")

plt.show()
