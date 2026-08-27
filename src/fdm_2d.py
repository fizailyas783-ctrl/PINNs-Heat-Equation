import numpy as np

# ------------------------------------------------------------
# Finite Difference Method (FDM) solver for the 2D Heat Equation
# u_t = alpha * (u_xx + u_yy)
#
# This extends the 1D FDM solver to two spatial dimensions,
# solving heat diffusion across a 2D square plate.
# ------------------------------------------------------------

def solve_heat_equation_2d_fdm(alpha=1.0, n=41):
    """
    Solves the 2D heat equation using the explicit finite
    difference method.

    Parameters:
        alpha : thermal diffusivity constant
        n     : number of grid points along each spatial axis (x and y)

    Returns:
        x, y : 1D arrays of spatial coordinates
        t    : 1D array of time points
        U    : 3D array of temperature values, shape (nt, n, n)
    """

    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Stability condition for 2D explicit FDM:
    # dt <= 1 / (2*alpha*(1/dx^2 + 1/dy^2))
    dt_max = 1.0 / (2 * alpha * (1 / dx**2 + 1 / dy**2))
    dt = 0.9 * dt_max
    t = np.arange(0, 1 + dt, dt)
    nt = len(t)

    # Initialize solution array: (time, x, y)
    U = np.zeros((nt, n, n))

    # Create meshgrid for initial condition
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Initial condition: u(x, y, 0) = sin(pi*x) * sin(pi*y)
    U[0, :, :] = np.sin(np.pi * X) * np.sin(np.pi * Y)

    # Boundary conditions: u = 0 on all four edges
    U[:, 0, :] = 0.0
    U[:, -1, :] = 0.0
    U[:, :, 0] = 0.0
    U[:, :, -1] = 0.0

    rx = alpha * dt / (dx ** 2)
    ry = alpha * dt / (dy ** 2)

    # ------------------------------------------------------------
    # Time-stepping loop
    # ------------------------------------------------------------
    for n_step in range(0, nt - 1):
        U_curr = U[n_step]

        laplacian = (
            rx * (U_curr[2:, 1:-1] - 2 * U_curr[1:-1, 1:-1] + U_curr[:-2, 1:-1]) +
            ry * (U_curr[1:-1, 2:] - 2 * U_curr[1:-1, 1:-1] + U_curr[1:-1, :-2])
        )

        U[n_step + 1, 1:-1, 1:-1] = U_curr[1:-1, 1:-1] + laplacian

        # Re-apply boundary conditions
        U[n_step + 1, 0, :] = 0.0
        U[n_step + 1, -1, :] = 0.0
        U[n_step + 1, :, 0] = 0.0
        U[n_step + 1, :, -1] = 0.0

    return x, y, t, U


if __name__ == "__main__":
    x, y, t, U = solve_heat_equation_2d_fdm(alpha=1.0)
    print(f"Grid shape: {U.shape}")
    print(f"Final time reached: {t[-1]:.4f}")
    print(f"Max temperature at t=0: {U[0].max():.4f}")
    print(f"Max temperature at final time: {U[-1].max():.4f}")
