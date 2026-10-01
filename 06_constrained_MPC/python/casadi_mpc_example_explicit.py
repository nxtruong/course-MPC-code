"""Constrained MPC with CasADi using the explicit (dense) prediction form.

This is a direct Python counterpart of ``yalmip_mpc_example_explicit.m``.
Install the packages in requirements.txt, then run this file directly.
"""

from __future__ import annotations

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np

from mpc_common import A, B, HORIZON, Q, R, plot_trajectories, simulate


def make_controller():
    """Build an MPC controller with inputs as the only decision trajectory."""
    nx, nu = B.shape
    opti = ca.Opti()
    x_current = opti.parameter(nx)
    u = opti.variable(nu, HORIZON)

    # These epigraph variables encode the absolute values in the 1-norm cost.
    state_magnitude = opti.variable(nx, HORIZON)
    input_magnitude = opti.variable(nu, HORIZON)

    x = x_current
    objective = 0
    for k in range(HORIZON):
        weighted_state = ca.DM(Q) @ x
        weighted_input = R * u[:, k]
        opti.subject_to(state_magnitude[:, k] >= weighted_state)
        opti.subject_to(state_magnitude[:, k] >= -weighted_state)
        opti.subject_to(input_magnitude[:, k] >= weighted_input)
        opti.subject_to(input_magnitude[:, k] >= -weighted_input)
        objective += ca.sum1(state_magnitude[:, k])
        objective += ca.sum1(input_magnitude[:, k])

        opti.subject_to(opti.bounded(-1.0, u[:, k], 1.0))
        x = ca.DM(A) @ x + ca.DM(B) @ u[:, k]
        # Constrain the successor state x_(k+1); x_0 is the given parameter.
        opti.subject_to(opti.bounded(-5.0, x, 5.0))

    opti.minimize(objective)
    opti.solver(
        "ipopt",
        {"print_time": False},
        {"print_level": 0, "sb": "yes", "tol": 1e-9},
    )

    def controller(measured_state: np.ndarray) -> np.ndarray:
        opti.set_value(x_current, measured_state)
        try:
            solution = opti.solve()
        except RuntimeError as error:
            raise RuntimeError(
                f"Explicit MPC solve failed for state {measured_state}."
            ) from error

        optimal_inputs = np.asarray(solution.value(u), dtype=float).reshape(-1)
        opti.set_initial(opti.x, solution.value(opti.x))
        return optimal_inputs

    return controller


def main():
    states, inputs = simulate(make_controller())
    print("Final state:", states[-1])
    print("Applied inputs:", inputs)
    plot_trajectories(states, inputs, "explicit/dense")
    plt.show()


if __name__ == "__main__":
    main()
