# Receding-Horizon LQR: Python Example

This example accompanies Lecture 5, **Optimal Control Introduction and
Unconstrained Linear Quadratic Control**, especially slides 46 and 54.
`RHLQR.py` and the self-contained `RHLQR.ipynb` simulate the discrete-time
plant from slide 46 and solve a new finite-horizon quadratic program at every sample. It applies only the first
optimized input before measuring the next state.

## Example

| Lecture topic | Example | Main ideas |
| --- | --- | --- |
| Impact of horizon length (slide 46) | `RHLQR.py` / `RHLQR.ipynb` | Batch CVXPY/OSQP optimization for horizons $N=2,\ldots,10$; closed-loop output responses |
| Choice of terminal weight (slide 54) | `RHLQR.py` / `RHLQR.ipynb` | Discrete algebraic Riccati equation (ARE); compare $N=5$ with $P=I$ or $P=P_\infty$ against $N=10$ with $P=I$ |

The model and objective are

$$
x_{k+1}=\begin{bmatrix}1.988&-0.998\\1&0\end{bmatrix}x_k
       +\begin{bmatrix}0.125\\0\end{bmatrix}u_k,
\qquad
J_N=\sum_{k=0}^{N-1}(x_k^\mathsf{T}Qx_k+u_k^\mathsf{T}Ru_k)
       +x_N^\mathsf{T}Px_N,
$$

with $Q=I$, $R=1$, and initially $P=I$. The script uses
$x_0=[0.4,\,0.4]^\mathsf{T}$ and displays the first state as the output.
It solves the ARE using `scipy.linalg.solve_discrete_are` and prints the
resulting matrix and equation residual.

## Setup

Create a virtual environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Run

From this directory, run the script:

```bash
python RHLQR.py
```

Or open `RHLQR.ipynb` in Jupyter, VS Code, or Google Colab and run its
cells in order. The notebook contains all its own code and does not import
the script. In Colab, the first code cell installs the required packages.

Both versions display two interactive figures. The first plots 150-step output
responses for $N=2,\ldots,10$ with $P=I$. The second plots 50-step output
and input responses for $N=5$ with $P=I$, $N=5$ with $P=P_\infty$, and
$N=10$ with $P=I$. This compares a longer horizon against a shorter horizon
with the ARE terminal cost. The horizontal axis counts samples (10 samples
per second), matching slide 46. The figures are displayed directly and are
not saved.
