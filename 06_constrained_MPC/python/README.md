# Constrained MPC: Python/CasADi Examples

These examples accompany Lecture 6, **Constrained Model Predictive Control**.
They translate the two MATLAB/YALMIP examples in `../matlab` to Python and
CasADi while retaining the original model, cost, bounds, horizon, initial
condition, simulation length, and receding-horizon implementation.

## Examples

| File | Prediction form | Optimization variables |
| --- | --- | --- |
| `casadi_mpc_example_explicit.py` | Explicit (dense) | The input trajectory; predicted states are symbolic expressions |
| `casadi_mpc_example_implicit.py` | Implicit (sparse) | The input and state trajectories; dynamics are equality constraints |
| `mpc_common.py` | Shared utilities | Plant data, closed-loop simulation, and plotting |

Both controllers solve

$$
x_{k+1}=\begin{bmatrix}2&-1\\1&0.2\end{bmatrix}x_k
       +\begin{bmatrix}1\\0\end{bmatrix}u_k,
\qquad
J_N=\sum_{k=0}^{N-1}\left(\lVert Qx_k\rVert_1
       +\lVert Ru_k\rVert_1\right),
$$

with $Q=I$, $R=2$, $N=7$, $-1\leq u_k\leq1$, and state bounds of
$-5\leq x_k\leq5$ for $k=1,\ldots,N$. The measured initial state $x_0$ is
a given parameter and is not subject to the state constraint. The 1-norm is
represented with epigraph variables, so the model supplied to IPOPT contains
only a linear objective and linear constraints. At each of 15 simulation
steps, the controller is solved for the measured state and only its first
optimized input is applied.

## Setup

Create a virtual environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.
CasADi wheels normally include the IPOPT solver used by these scripts.

## Run

From this directory, run either example:

```bash
python casadi_mpc_example_explicit.py
python casadi_mpc_example_implicit.py
```

Each script prints the applied input sequence and final state, then displays
a figure with the closed-loop state and input trajectories. Figures are not
saved to disk.
