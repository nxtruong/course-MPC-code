"""Receding-horizon LQR example from Lecture 5, slides 46 and 54.

Run with ``python RHLQR.py`` after installing numpy, scipy, matplotlib,
cvxpy, and osqp. Figures are displayed interactively and are not saved.

The horizon N here means N control moves, as in the lecture's written cost
sum_{k=0}^{N-1} (x_k' Q x_k + u_k' R u_k) + x_N' P x_N.
"""

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_discrete_are


# Slide 46's discrete-time model and quadratic weights.
A = np.array([[1.988, -0.998], [1.0, 0.0]])
B = np.array([[0.125], [0.0]])
Q = np.eye(2)
R = np.eye(1)
P_IDENTITY = np.eye(2)
X_INITIAL = np.array([0.4, 0.4])  # Reproduces the plotted initial output.
HORIZON_SIMULATION_STEPS = 150
TERMINAL_SIMULATION_STEPS = 50


def make_batch_problem(horizon: int, terminal_weight: np.ndarray):
    """Build the full finite-horizon QP once for repeated MPC solves.

    X and U contain all predicted states and inputs. The measured state is
    supplied through a CVXPY Parameter at each sampling instant.
    """
    if horizon < 1:
        raise ValueError("horizon must contain at least one control move")

    x_measured = cp.Parameter(2)
    x = cp.Variable((2, horizon + 1))
    u = cp.Variable((1, horizon))
    constraints = [x[:, 0] == x_measured]
    cost = 0
    for k in range(horizon):
        constraints.append(x[:, k + 1] == A @ x[:, k] + B @ u[:, k])
        cost += cp.quad_form(x[:, k], Q) + cp.quad_form(u[:, k], R)
    cost += cp.quad_form(x[:, horizon], terminal_weight)
    problem = cp.Problem(cp.Minimize(cost), constraints)
    return problem, x_measured, u


def simulate_receding_horizon(horizon: int, terminal_weight: np.ndarray,
                             simulation_steps: int):
    """Resolve the batch QP, apply its first input, and advance the plant."""
    if simulation_steps < 1:
        raise ValueError("simulation_steps must be positive")
    problem, x_measured, u = make_batch_problem(horizon, terminal_weight)
    states = np.empty((simulation_steps + 1, 2))
    inputs = np.empty(simulation_steps)
    states[0] = X_INITIAL

    for k in range(simulation_steps):
        x_measured.value = states[k]
        problem.solve(solver=cp.OSQP, warm_start=True, eps_abs=1e-8,
                      eps_rel=1e-8)
        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(
                f"OSQP failed at step {k}, horizon {horizon}: {problem.status}"
            )
        inputs[k] = u.value[0, 0]
        states[k + 1] = A @ states[k] + B[:, 0] * inputs[k]

    return states, inputs


def plot_horizon_comparison(responses: dict[int, tuple[np.ndarray, np.ndarray]]):
    """Plot x_1, the output shown in slide 46, for horizons 2 through 10."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for horizon, (states, _) in responses.items():
        ax.plot(np.arange(len(states)), states[:, 0],
                label=f"N = {horizon}", linewidth=1.7)
    ax.set(title="Closed-loop response: impact of horizon length ($P=Q=I$, $R=1$)",
           xlabel="Time (samples)", ylabel="Output $x_1$")
    ax.set(xlim=(0, HORIZON_SIMULATION_STEPS), ylim=(-0.4, 0.4))
    ax.grid(True, alpha=0.35)
    ax.legend(ncol=3)
    fig.tight_layout()


def plot_terminal_comparison(responses):
    """Compare a short ARE horizon with short and long identity-cost horizons."""
    fig, (ax_y, ax_u) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for label, (states, inputs) in responses.items():
        ax_y.plot(np.arange(len(states)), states[:, 0],
                  label=label, linewidth=1.8)
        ax_u.step(np.arange(len(inputs)), inputs, where="post",
                  label=label, linewidth=1.4)
    ax_y.set(ylabel="Output $x_1$",
             title="Terminal cost and horizon comparison (50 steps)")
    ax_u.set(xlabel="Time (samples)", ylabel="Input $u$")
    ax_u.set_xlim(0, TERMINAL_SIMULATION_STEPS)
    for ax in (ax_y, ax_u):
        ax.grid(True, alpha=0.35)
        ax.legend()
    fig.tight_layout()


def main():
    responses = {
        horizon: simulate_receding_horizon(
            horizon, P_IDENTITY, HORIZON_SIMULATION_STEPS)
        for horizon in range(2, 11)
    }
    plot_horizon_comparison(responses)

    # Slide 54: P = A' P A + Q - A' P B (R + B' P B)^(-1) B' P A.
    p_are = solve_discrete_are(A, B, Q, R)
    print("Discrete-time ARE solution P =\n", p_are)
    print("ARE residual (Frobenius norm):", np.linalg.norm(
        p_are - (A.T @ p_are @ A + Q
                 - A.T @ p_are @ B
                 @ np.linalg.solve(R + B.T @ p_are @ B, B.T @ p_are @ A))
    ))

    terminal_responses = {
        "$N=5$, $P=I$": simulate_receding_horizon(
            5, P_IDENTITY, TERMINAL_SIMULATION_STEPS),
        "$N=5$, $P=P_{\\infty}$ (ARE)": simulate_receding_horizon(
            5, p_are, TERMINAL_SIMULATION_STEPS),
        "$N=10$, $P=I$": simulate_receding_horizon(
            10, P_IDENTITY, TERMINAL_SIMULATION_STEPS),
    }
    plot_terminal_comparison(terminal_responses)
    plt.show()


if __name__ == "__main__":
    main()
