import torch
import torch.nn as nn

# ------------------------------------------------------------
# This class defines our neural network for the PINN.
# It takes (x, t) as input and predicts u(x, t) as output.
# ------------------------------------------------------------
class PINN(nn.Module):
    def __init__(self, hidden_layers=4, neurons=20):
        super(PINN, self).__init__()

        # Input layer: 2 inputs (x, t) -> first hidden layer
        layers = [nn.Linear(2, neurons), nn.Tanh()]

        # Add a few hidden layers (small network, beginner-friendly)
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(neurons, neurons))
            layers.append(nn.Tanh())

        # Output layer: last hidden layer -> 1 output (u)
        layers.append(nn.Linear(neurons, 1))

        # Combine all layers into a single sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x, t):
        # Combine x and t into a single input tensor: shape (N, 2)
        inputs = torch.cat([x, t], dim=1)
        u = self.model(inputs)
        return u
