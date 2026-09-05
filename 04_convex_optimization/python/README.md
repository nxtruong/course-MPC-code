# Convex Optimization: Python Tools

These examples accompany Lecture 4, **Convex Optimization**. They use small,
deterministic problems to show the difference between calling a solver directly
and describing a problem through a modeling tool.

## Examples

| Lecture topic | Example | Main ideas |
| --- | --- | --- |
| Linear and Quadratic Programs; Optimality Conditions | `osqp_direct` | Direct sparse QP data, solver status, primal and dual solutions, KKT residuals, infeasibility certificates, and repeated solves |
| Linear and Quadratic Programs; Practical Optimization | `cvxpy_modeling` | Readable LP/QP models, DCP checking, solver selection, explicit-constraint duals, sensitivity, and status handling |
| Practical Optimization; Optimality Conditions | `casadi_modeling` | CasADi Opti models for a QP and a small convex NLP, automatic derivatives, multipliers, and IPOPT diagnostics |
| What might go wrong? | All three | Infeasible and unbounded problems, status-first result handling, and the distinction between diagnostics and certificates |

Each topic has a guided Jupyter notebook and an equivalent Python script:

- `osqp_direct.ipynb` / `osqp_direct.py`
- `cvxpy_modeling.ipynb` / `cvxpy_modeling.py`
- `casadi_modeling.ipynb` / `casadi_modeling.py`

The repeated QP solves illustrate the setup/update pattern used in online MPC,
but these examples do not formulate a complete MPC controller.

## Setup

Create a virtual environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

CasADi wheels normally include qpOASES and IPOPT. If either plugin is absent,
the CasADi script reports which bundled capability is required rather than
silently changing algorithms.

## Run

Open any notebook in Jupyter, VS Code, or Google Colab and run its cells in
order. Each notebook contains all its code and numerical checks; no companion
`.py` file is needed. In Colab, upload only the notebook and uncomment and run
the optional package-installation command if its dependencies are missing.

The scripts can be run from this directory:

```bash
python osqp_direct.py
python cvxpy_modeling.py
python casadi_modeling.py
```

Solver iteration logs are disabled by default. Each script prints a compact
summary and contains assertions against analytical solutions and numerical
certificate conditions.

## Interpretation notes

- The lecture writes inequalities as `g(x) <= 0` with nonnegative multipliers.
  Constraint rows in the examples are deliberately ordered so returned duals
  can be compared with the lecture's KKT example.
- OSQP accepts `l <= A x <= u`. Multipliers for active upper bounds are positive;
  multipliers for active lower bounds are negative. The principal QP uses only
  upper-bound rows so its multipliers directly match the lecture convention.
- CVXPY records dual values for explicit constraints. A sign or bound supplied
  only as a variable attribute does not produce a separate dual value.
- KKT residuals and solver certificates are checked numerically, not symbolically.
- CVXPY verifies disciplined convex programming rules. CasADi/IPOPT is a general
  local nonlinear optimization framework; the modeler remains responsible for
  recognizing when a problem is convex.
