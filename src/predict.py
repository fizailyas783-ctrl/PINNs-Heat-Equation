import torch
import numpy as np
import matplotlib.pyplot as plt
from model import PINN
from fdm_solver import solve_heat_equation_fdm

# ------------------------------------------------------------
# Settings (must match train.py)
# ------------------------------------------------------------
alpha = 1.0

# ------------------------------------------------------------
# Load the trained PINN model
# ------------------------------------------------------------
model = PINN(hidden_layers=4, neurons=20)
model.load_state_dict(torch.load("trained_model.pth"))
model.eval()

# ------------------------------------------------------------
# Create a grid of (x, t) points to evaluate the PINN and exact solution
# ------------------------------------------------------------
n_points = 100
x_vals = np.linspace(0, 1, n_points)
t_vals = np.linspace(0, 1, n_points)

X, T = np.meshgrid(x_vals, t_vals)

x_flat = torch.tensor(X.flatten(), dtype=torch.float32).view(-1, 1)
t_flat = torch.tensor(T.flatten(), dtype=torch.float32).view(-1, 1)

# ------------------------------------------------------------
# PINN prediction
# ------------------------------------------------------------
with torch.no_grad():
    u_pinn = model(x_flat, t_flat).numpy().reshape(n_points, n_points)

# ------------------------------------------------------------
# Exact analytical solution
# u(x,t) = sin(pi*x) * exp(-pi^2 * alpha * t)
# ------------------------------------------------------------
u_exact = np.sin(np.pi * X) * np.exp(-(np.pi**2) * alpha * T)

# ------------------------------------------------------------
# Finite Difference Method (FDM) solution
# Note: FDM uses its own grid (different resolution/time steps),
# so we interpolate it onto the same (x_vals, t_vals) grid used
# by the PINN and exact solution, to allow a fair comparison.
# ------------------------------------------------------------
x_fdm, t_fdm, U_fdm = solve_heat_equation_fdm(alpha=alpha, nx=101)

from scipy.interpolate import RectBivariateSpline
fdm_interpolator = RectBivariateSpline(t_fdm, x_fdm, U_fdm)
u_fdm = fdm_interpolator(t_vals, x_vals)

# ------------------------------------------------------------
# Error metrics: compare PINN and FDM against the exact solution
# ------------------------------------------------------------
def compute_errors(u_approx, u_true):
    error = u_approx - u_true
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error**2))
    max_err = np.max(np.abs(error))
    return mae, rmse, max_err

mae_pinn, rmse_pinn, max_pinn = compute_errors(u_pinn, u_exact)
mae_fdm, rmse_fdm, max_fdm = compute_errors(u_fdm, u_exact)

print("\n===== Error Analysis (compared to Exact Solution) =====")
print(f"{'Method':<20}{'MAE':<15}{'RMSE':<15}{'Max Error':<15}")
print(f"{'PINN':<20}{mae_pinn:<15.6f}{rmse_pinn:<15.6f}{max_pinn:<15.6f}")
print(f"{'Finite Difference':<20}{mae_fdm:<15.6f}{rmse_fdm:<15.6f}{max_fdm:<15.6f}")
print("=========================================================\n")

# ------------------------------------------------------------
# Plot: PINN vs FDM vs Exact solution (three heatmaps)
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

im0 = axes[0].pcolormesh(X, T, u_pinn, shading='auto', cmap='hot')
axes[0].set_title("PINN Prediction")
axes[0].set_xlabel("x")
axes[0].set_ylabel("t")
fig.colorbar(im0, ax=axes[0])

im1 = axes[1].pcolormesh(X, T, u_fdm, shading='auto', cmap='hot')
axes[1].set_title("Finite Difference Method")
axes[1].set_xlabel("x")
axes[1].set_ylabel("t")
fig.colorbar(im1, ax=axes[1])

im2 = axes[2].pcolormesh(X, T, u_exact, shading='auto', cmap='hot')
axes[2].set_title("Exact Solution")
axes[2].set_xlabel("x")
axes[2].set_ylabel("t")
fig.colorbar(im2, ax=axes[2])

plt.tight_layout()

# ------------------------------------------------------------
# Save the figure into the results folder
# ------------------------------------------------------------
plt.savefig("../results/heat_solution.png", dpi=150)
print("Plot saved to results/heat_solution.png")

plt.show()
