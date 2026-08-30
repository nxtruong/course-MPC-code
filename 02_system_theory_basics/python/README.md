# System Theory Basics: Python Examples

Python code for the lecture on System Theory Basics: the nonlinear pendulum and
the discrete-time Lyapunov equation.

## Setup

From this directory, create a virtual environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Run

Open `pendulum.ipynb`, `pendulum_casadi.ipynb`, or `lyap.ipynb` in
Jupyter, VS Code, or Google Colab and run the cells in order. Each notebook
begins with an optional Colab setup cell that installs its required libraries.

The equivalent scripts can be run from a terminal:

```bash
python pendulum.py
python pendulum_casadi.py
python lyap.py
```

`pendulum.py` displays the nonlinear response, the nonlinear/linearized
closed-loop comparison, and the continuous/discrete model comparison.
`pendulum_casadi.py` repeats these experiments with CasADi integrators and
compares autodiff Jacobians with their analytical values.
`lyap.py` prints only the solution matrix `P`.
