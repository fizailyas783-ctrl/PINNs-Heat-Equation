import torch
import numpy as np
from model import PINN

# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------
torch.manual_seed(42)          # for reproducible results
alpha = 1.0                    # thermal diffusivity constant
epochs = 5000                  # number of training iterations
lr = 1e-3                      # learning rate

# Number of random sample points for each loss term
N_pde = 2000     # points inside the domain (for PDE residual)
N_ic = 200       # points on the initial condition (t = 0)
N_bc = 200       # points on the boundaries (x = 0 and x = 1)

# ------------------------------------------------------------
# Create the model
# ------------------------------------------------------------
model = PINN(hidden_layers=4, neurons=20)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# ------------------------------------------------------------
# Function: PDE residual loss
# R(x,t) = u_t - alpha * u_xx   should be close to 0
# ------------------------------------------------------------
def pde_loss():
    x = torch.rand(N_pde, 1, requires_grad=True)
    t = torch.rand(N_pde, 1, requires_grad=True)

    u = model(x, t)

    # First derivative: du/dt
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                               create_graph=True)[0]

    # First derivative: du/dx
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                               create_graph=True)[0]

    # Second derivative: d^2u/dx^2
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                create_graph=True)[0]

    residual = u_t - alpha * u_xx
    return torch.mean(residual**2)

# ------------------------------------------------------------
# Function: Initial condition loss
# u(x, 0) = sin(pi * x)
# ------------------------------------------------------------
def ic_loss():
    x = torch.rand(N_ic, 1)
    t = torch.zeros(N_ic, 1)   # t = 0

    u_pred = model(x, t)
    u_true = torch.sin(np.pi * x)

    return torch.mean((u_pred - u_true)**2)

# ------------------------------------------------------------
# Function: Boundary condition loss
# u(0, t) = 0  and  u(1, t) = 0
# ------------------------------------------------------------
def bc_loss():
    t = torch.rand(N_bc, 1)

    x0 = torch.zeros(N_bc, 1)  # x = 0
    x1 = torch.ones(N_bc, 1)   # x = 1

    u0 = model(x0, t)
    u1 = model(x1, t)

    return torch.mean(u0**2) + torch.mean(u1**2)

# ------------------------------------------------------------
# Training loop
# ------------------------------------------------------------
print("Starting training...")

for epoch in range(epochs):
    optimizer.zero_grad()

    loss_pde = pde_loss()
    loss_ic = ic_loss()
    loss_bc = bc_loss()

    # Total loss = sum of all three parts
    loss = loss_pde + loss_ic + loss_bc

    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch:5d} | Total Loss: {loss.item():.6f} "
              f"| PDE: {loss_pde.item():.6f} "
              f"| IC: {loss_ic.item():.6f} "
              f"| BC: {loss_bc.item():.6f}")

print("Training finished.")

# ------------------------------------------------------------
# Save the trained model so predict.py can use it
# ------------------------------------------------------------
torch.save(model.state_dict(), "trained_model.pth")
print("Model saved as trained_model.pth")
