import torch
import torch.nn as nn

# ------------------------------------------------------------
# Neural network for the 2D Heat Equation PINN.
# Takes (x, y, t) as input and predicts u(x, y, t).
#
# This extends the 1D model by adding a second spatial
# dimension (y), so the network now has 3 inputs instead of 2.
# ------------------------------------------------------------
class PINN2D(nn.Module):
    def __init__(self, hidden_layers=4, neurons=30):
        super(PINN2D, self).__init__()

        # Input layer: 3 inputs (x, y, t) -> first hidden layer
        layers = [nn.Linear(3, neurons), nn.Tanh()]

        # Hidden layers
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(neurons, neurons))
            layers.append(nn.Tanh())

        # Output layer: last hidden layer -> 1 output (u)
        layers.append(nn.Linear(neurons, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x, y, t):
        # Combine x, y and t into a single input tensor: shape (N, 3)
        inputs = torch.cat([x, y, t], dim=1)
        u = self.model(inputs)
        return u
