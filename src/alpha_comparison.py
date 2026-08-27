import numpy as np
import matplotlib.pyplot as plt
from fdm_solver import solve_heat_equation_fdm

# ------------------------------------------------------------
# This script compares the heat equation's behavior for
# different values of the thermal diffusivity constant (alpha).
#
# A smaller alpha means heat spreads/decays more slowly.
# A larger alpha means heat spreads/decays more quickly.
# ------------------------------------------------------------

alpha_values = [0.1, 0.5, 1.0]
colors = ['blue', 'green', 'red']

# We will look at the temperature profile at a fixed time slice
# to clearly see the effect of alpha.
fixed_time = 0.2

plt.figure(figsize=(8, 5))

for alpha, color in zip(alpha_values, colors):
    x, t, U = solve_heat_equation_fdm(alpha=alpha, nx=101)

    # Find the index of the time point closest to fixed_time
    time_index = np.argmin(np.abs(t - fixed_time))

    plt.plot(x, U[time_index, :], color=color,
              label=f"alpha = {alpha}")

plt.title(f"Temperature Profile at t = {fixed_time} for Different Alpha Values")
plt.xlabel("x")
plt.ylabel("u(x, t)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("../results/alpha_comparison.png", dpi=150)
print("Plot saved to results/alpha_comparison.png")
plt.show()
