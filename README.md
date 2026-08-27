# Physics-Informed Neural Network (PINN) for the 1D Heat Equation

## Project Overview

This project implements a Physics-Informed Neural Network (PINN) to approximate the solution of the 1D Heat Equation. Instead of learning from labeled data, the neural network learns by satisfying the governing partial differential equation (PDE) directly, along with the initial and boundary conditions.

This is a small, beginner-friendly project built to demonstrate the connection between mathematics, PDEs, scientific computing, and machine learning.

## Mathematical Equation

The 1D Heat Equation: u_t = alpha * u_xx

where:
- u(x, t) is the temperature at position x and time t
- alpha is the thermal diffusivity constant (set to 1.0 in this project)

Domain used: 0 <= x <= 1, 0 <= t <= 1

## Initial and Boundary Conditions

- Initial condition: u(x, 0) = sin(pi * x)
- Boundary conditions: u(0, t) = 0 and u(1, t) = 0

These conditions were chosen because they lead to a simple, well-known analytical solution, which allows the PINN's prediction to be validated against an exact mathematical answer: u(x, t) = sin(pi * x) * exp(-pi^2 * alpha * t)

## What is a PINN?

A Physics-Informed Neural Network is a neural network trained to satisfy a physical law (expressed as a differential equation) rather than being trained only on labeled data. The network takes (x, t) as input and predicts u(x, t). Using automatic differentiation, the derivatives u_t and u_xx are computed directly from the network's output, and the PDE residual is calculated and minimized during training.

## Methodology

1. The neural network takes (x, t) as input and outputs a predicted temperature u(x, t).
2. Automatic differentiation is used to compute u_t (first derivative in time) and u_xx (second derivative in space).
3. The PDE residual is calculated as: R(x, t) = u_t - alpha * u_xx
4. Three loss terms are combined:
   - PDE loss: how well the network satisfies the heat equation at random interior points
   - Initial condition loss: how well the network matches u(x, 0) = sin(pi * x)
   - Boundary condition loss: how well the network satisfies u(0, t) = 0 and u(1, t) = 0
5. Total loss: Loss = Loss_PDE + Loss_initial + Loss_boundary
6. The network is trained using the Adam optimizer to minimize this total loss.

## Project Structure

PINNs-Heat-Equation/
- README.md
- requirements.txt
- src/
  - model.py (Neural network architecture)
  - train.py (Training loop and loss functions)
  - predict.py (Generates prediction and comparison plot)
- results/
  - heat_solution.png

## Installation

Clone the repository and install the required libraries:

git clone https://github.com/fizailyas783-ctrl/PINNs-Heat-Equation.git
cd PINNs-Heat-Equation
pip install -r requirements.txt

## How to Run

Navigate to the src folder and run the training script, followed by the prediction script:

cd src
python train.py
python predict.py

This project can also be run directly in Google Colab without any local installation.

After running, the trained model is saved as trained_model.pth, and the resulting plot is saved to results/heat_solution.png.

## Results

The image below shows a side-by-side comparison of the PINN's predicted solution and the exact analytical solution:

![Heat Equation Solution](results/heat_solution.png)

Both plots show the temperature u(x, t) across position and time. The close match between the PINN prediction and the exact solution confirms that the network successfully learned the underlying physics of the heat equation.

## Mathematical Explanation

- PDE Residual: Measures how far the network's output is from satisfying the heat equation. A residual of zero means the equation is perfectly satisfied.
- Automatic Differentiation: Allows exact computation of derivatives (u_t, u_xx) directly from the neural network, without needing numerical approximation methods like finite differences.
- Initial Condition: Anchors the solution at the starting time (t = 0), preventing the network from learning trivial or incorrect solutions.
- Boundary Conditions: Constrain the solution's behavior at the edges of the spatial domain, ensuring physical consistency.

## Technologies Used

- Python
- PyTorch
- NumPy
- Matplotlib

## Future Improvements

- Extend to 2D or 3D heat equations
- Experiment with different initial/boundary conditions
- Add a Streamlit web app for interactive visualization
- Compare performance against traditional numerical solvers (e.g., finite difference methods)

## Limitations

- This project uses a simple domain and simple conditions for educational clarity.
- It is trained on a single fixed value of alpha.
- It is designed as a learning project, not for production-scale physics simulation.
