"""Constrained MPC with CasADi using the implicit (sparse) prediction form.

This is a direct Python counterpart of ``yalmip_mpc_example_implicit.m``.
Install the packages in requirements.txt, then run this file directly.
"""

from __future__ import annotations

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np

from mpc_common import A, B, HORIZON, Q, R, plot_trajectories, simulate


def make_controller():
    """Build an MPC controller whose states and inputs are decision variables."""
    nx, nu = B.shape
    opti = ca.Opti()
    x_current = opti.parameter(nx)
    x = opti.variable(nx, HORIZON + 1)
    u = opti.variable(nu, HORIZON)

    # Epigraph variables give an exact, differentiable linear formulation of
    # the 1-norm objective used in the YALMIP example.
    state_magnitude = opti.variable(nx, HORIZON)
    input_magnitude = opti.variable(nu, HORIZON)

    opti.subject_to(x[:, 0] == x_current)
    objective = 0
    for k in range(HORIZON):
        weighted_state = ca.DM(Q) @ x[:, k]
        weighted_input = R * u[:, k]
        opti.subject_to(state_magnitude[:, k] >= weighted_state)
        opti.subject_to(state_magnitude[:, k] >= -weighted_state)
        opti.subject_to(input_magnitude[:, k] >= weighted_input)
        opti.subject_to(input_magnitude[:, k] >= -weighted_input)
        objective += ca.sum1(state_magnitude[:, k])
        objective += ca.sum1(input_magnitude[:, k])

        opti.subject_to(x[:, k + 1] == ca.DM(A) @ x[:, k] + ca.DM(B) @ u[:, k])
        opti.subject_to(opti.bounded(-1.0, u[:, k], 1.0))
        # Constrain x_1,...,x_N; x_0 is the given parameter.
        opti.subject_to(opti.bounded(-5.0, x[:, k + 1], 5.0))

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
                f"Implicit MPC solve failed for state {measured_state}."
            ) from error

        optimal_inputs = np.asarray(solution.value(u), dtype=float).reshape(-1)
        # Reuse the previous solution as the next IPOPT initial guess.
        opti.set_initial(opti.x, solution.value(opti.x))
        return optimal_inputs

    return controller


def main():
    states, inputs = simulate(make_controller())
    print("Final state:", states[-1])
    print("Applied inputs:", inputs)
    plot_trajectories(states, inputs, "implicit/sparse")
    plt.show()


if __name__ == "__main__":
    main()
