import torch
import numpy as np
import matplotlib.pyplot as plt
from model_2d import PINN2D
from fdm_2d import solve_heat_equation_2d_fdm
from scipy.interpolate import RegularGridInterpolator

# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------
alpha = 1.0
fixed_time = 0.2   # snapshot time to visualize

# ------------------------------------------------------------
# Load the trained 2D PINN model
# ------------------------------------------------------------
model = PINN2D(hidden_layers=4, neurons=30)
model.load_state_dict(torch.load("trained_model_2d.pth"))
model.eval()

# ------------------------------------------------------------
# Create a spatial grid at the fixed time
# ------------------------------------------------------------
n_points = 50
x_vals = np.linspace(0, 1, n_points)
y_vals = np.linspace(0, 1, n_points)
X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')

x_flat = torch.tensor(X.flatten(), dtype=torch.float32).view(-1, 1)
y_flat = torch.tensor(Y.flatten(), dtype=torch.float32).view(-1, 1)
t_flat = torch.full_like(x_flat, fixed_time)

# ------------------------------------------------------------
# PINN prediction at fixed_time
# ------------------------------------------------------------
with torch.no_grad():
    u_pinn = model(x_flat, y_flat, t_flat).numpy().reshape(n_points, n_points)

# ------------------------------------------------------------
# Exact analytical solution at fixed_time
# u(x,y,t) = sin(pi*x) * sin(pi*y) * exp(-2*pi^2*alpha*t)
# ------------------------------------------------------------
u_exact = (np.sin(np.pi * X) * np.sin(np.pi * Y) *
           np.exp(-2 * (np.pi ** 2) * alpha * fixed_time))

# ------------------------------------------------------------
# FDM solution at fixed_time (interpolated onto the same grid)
# ------------------------------------------------------------
x_fdm, y_fdm, t_fdm, U_fdm = solve_heat_equation_2d_fdm(alpha=alpha, n=41)

time_index = np.argmin(np.abs(t_fdm - fixed_time))
U_slice = U_fdm[time_index]   # shape (n, n) at that time

interpolator = RegularGridInterpolator((x_fdm, y_fdm), U_slice)
points = np.stack([X.flatten(), Y.flatten()], axis=-1)
u_fdm = interpolator(points).reshape(n_points, n_points)

# ------------------------------------------------------------
# Error metrics
# ------------------------------------------------------------
def compute_errors(u_approx, u_true):
    error = u_approx - u_true
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error**2))
    max_err = np.max(np.abs(error))
    return mae, rmse, max_err

mae_pinn, rmse_pinn, max_pinn = compute_errors(u_pinn, u_exact)
mae_fdm, rmse_fdm, max_fdm = compute_errors(u_fdm, u_exact)

print(f"\n===== 2D Error Analysis at t = {fixed_time} (compared to Exact Solution) =====")
print(f"{'Method':<20}{'MAE':<15}{'RMSE':<15}{'Max Error':<15}")
print(f"{'PINN':<20}{mae_pinn:<15.6f}{rmse_pinn:<15.6f}{max_pinn:<15.6f}")
print(f"{'Finite Difference':<20}{mae_fdm:<15.6f}{rmse_fdm:<15.6f}{max_fdm:<15.6f}")
print("===============================================================\n")

# ------------------------------------------------------------
# Plot: PINN vs FDM vs Exact (2D heatmaps at fixed time)
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

im0 = axes[0].pcolormesh(X, Y, u_pinn, shading='auto', cmap='hot')
axes[0].set_title(f"PINN Prediction (t={fixed_time})")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
fig.colorbar(im0, ax=axes[0])

im1 = axes[1].pcolormesh(X, Y, u_fdm, shading='auto', cmap='hot')
axes[1].set_title(f"Finite Difference (t={fixed_time})")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
fig.colorbar(im1, ax=axes[1])

im2 = axes[2].pcolormesh(X, Y, u_exact, shading='auto', cmap='hot')
axes[2].set_title(f"Exact Solution (t={fixed_time})")
axes[2].set_xlabel("x")
axes[2].set_ylabel("y")
fig.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.savefig("../results/heat_solution_2d.png", dpi=150)
print("Plot saved to results/heat_solution_2d.png")
plt.show()
