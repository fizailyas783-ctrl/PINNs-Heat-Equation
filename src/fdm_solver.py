import numpy as np

# ------------------------------------------------------------
# Finite Difference Method (FDM) solver for the 1D Heat Equation
# u_t = alpha * u_xx
#
# This is a classical numerical method (no machine learning).
# It solves the PDE by discretizing space and time into a grid
# and stepping forward in time using an explicit scheme.
# ------------------------------------------------------------

def solve_heat_equation_fdm(alpha=1.0, nx=101, nt=5000):
    """
    Solves the 1D heat equation using the explicit finite
    difference method.

    Parameters:
        alpha : thermal diffusivity constant
        nx    : number of spatial grid points (0 to 1)
        nt    : number of time steps (0 to 1)

    Returns:
        x   : array of spatial points
        t   : array of time points
        U   : 2D array of temperature values, shape (nt, nx)
    """

    # Spatial grid
    x = np.linspace(0, 1, nx)
    dx = x[1] - x[0]

    # Time grid
    # Stability condition for explicit FDM: dt <= dx^2 / (2*alpha)
    dt_max = (dx ** 2) / (2 * alpha)
    dt = 0.9 * dt_max   # use 90% of max stable step size for safety
    t = np.arange(0, 1 + dt, dt)
    nt = len(t)

    # Initialize solution array: rows = time, columns = space
    U = np.zeros((nt, nx))

    # Initial condition: u(x, 0) = sin(pi * x)
    U[0, :] = np.sin(np.pi * x)

    # Boundary conditions: u(0, t) = 0, u(1, t) = 0
    U[:, 0] = 0.0
    U[:, -1] = 0.0

    # Coefficient used in the update formula
    r = alpha * dt / (dx ** 2)

    # ------------------------------------------------------------
    # Time-stepping loop: update temperature at each interior point
    # using the explicit finite difference formula:
    # U_new[i] = U[i] + r * (U[i+1] - 2*U[i] + U[i-1])
    # ------------------------------------------------------------
    for n in range(0, nt - 1):
        U[n + 1, 1:-1] = U[n, 1:-1] + r * (
            U[n, 2:] - 2 * U[n, 1:-1] + U[n, :-2]
        )
        # Re-apply boundary conditions (they should stay 0)
        U[n + 1, 0] = 0.0
        U[n + 1, -1] = 0.0

    return x, t, U


if __name__ == "__main__":
    # Quick test when running this file directly
    x, t, U = solve_heat_equation_fdm(alpha=1.0)
    print(f"Grid shape: {U.shape}")
    print(f"Final time reached: {t[-1]:.4f}")
    print(f"Max temperature at t=0: {U[0].max():.4f}")
    print(f"Max temperature at final time: {U[-1].max():.4f}")
