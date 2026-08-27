import torch
import numpy as np
from model_2d import PINN2D

# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------
torch.manual_seed(42)
alpha = 1.0
epochs = 5000
lr = 1e-3

N_pde = 3000     # interior points for PDE residual
N_ic = 300       # points for initial condition (t = 0)
N_bc = 300       # points per boundary edge

# ------------------------------------------------------------
# Create the model
# ------------------------------------------------------------
model = PINN2D(hidden_layers=4, neurons=30)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# ------------------------------------------------------------
# PDE residual loss: R = u_t - alpha * (u_xx + u_yy)
# ------------------------------------------------------------
def pde_loss():
    x = torch.rand(N_pde, 1, requires_grad=True)
    y = torch.rand(N_pde, 1, requires_grad=True)
    t = torch.rand(N_pde, 1, requires_grad=True)

    u = model(x, y, t)

    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                               create_graph=True)[0]

    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                               create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                create_graph=True)[0]

    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u),
                               create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y),
                                create_graph=True)[0]

    residual = u_t - alpha * (u_xx + u_yy)
    return torch.mean(residual**2)

# ------------------------------------------------------------
# Initial condition loss: u(x, y, 0) = sin(pi*x) * sin(pi*y)
# ------------------------------------------------------------
def ic_loss():
    x = torch.rand(N_ic, 1)
    y = torch.rand(N_ic, 1)
    t = torch.zeros(N_ic, 1)

    u_pred = model(x, y, t)
    u_true = torch.sin(np.pi * x) * torch.sin(np.pi * y)

    return torch.mean((u_pred - u_true)**2)

# ------------------------------------------------------------
# Boundary condition loss: u = 0 on all four edges
# (x=0, x=1, y=0, y=1)
# ------------------------------------------------------------
def bc_loss():
    t = torch.rand(N_bc, 1)

    # x = 0 edge
    y1 = torch.rand(N_bc, 1)
    u_x0 = model(torch.zeros(N_bc, 1), y1, t)

    # x = 1 edge
    y2 = torch.rand(N_bc, 1)
    u_x1 = model(torch.ones(N_bc, 1), y2, t)

    # y = 0 edge
    x1 = torch.rand(N_bc, 1)
    u_y0 = model(x1, torch.zeros(N_bc, 1), t)

    # y = 1 edge
    x2 = torch.rand(N_bc, 1)
    u_y1 = model(x2, torch.ones(N_bc, 1), t)

    return (torch.mean(u_x0**2) + torch.mean(u_x1**2) +
            torch.mean(u_y0**2) + torch.mean(u_y1**2))

# ------------------------------------------------------------
# Training loop
# ------------------------------------------------------------
print("Starting 2D training...")

for epoch in range(epochs):
    optimizer.zero_grad()

    loss_pde = pde_loss()
    loss_ic = ic_loss()
    loss_bc = bc_loss()

    loss = loss_pde + loss_ic + loss_bc

    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch:5d} | Total Loss: {loss.item():.6f} "
              f"| PDE: {loss_pde.item():.6f} "
              f"| IC: {loss_ic.item():.6f} "
              f"| BC: {loss_bc.item():.6f}")

print("Training finished.")

torch.save(model.state_dict(), "trained_model_2d.pth")
print("Model saved as trained_model_2d.pth")
