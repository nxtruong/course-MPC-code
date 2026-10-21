"""DC motor reference tracking.

Install dependencies with:
    python -m pip install numpy scipy matplotlib cvxpy osqp

The default configuration: N=4, LQR terminal cost,
and the physical voltage constraint 0 <= u <= 20 V. Try the slide cases:
    python tracking_mpc.py --horizon 4 --terminal-cost none --unconstrained
    python tracking_mpc.py --horizon 4 --terminal-cost lqr --unconstrained
    python tracking_mpc.py --horizon 20 --terminal-cost none --unconstrained
    python tracking_mpc.py --horizon 4 --terminal-cost lqr
    python tracking_mpc.py --horizon 20 --terminal-cost none
Use --horizon 10 for the other long-horizon example. Plots are displayed
interactively and are not saved. This example uses nominal tracking with
full-state feedback; it does not add a disturbance estimator or integrator.
"""

import argparse

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.signal import cont2discrete


# State x = [angular speed (rad/s), armature current (A)], input u = voltage (V).
A = np.array([[-10.0, 1.0], [-0.02, -2.0]])
B = np.array([[0.0], [2.0]])
C = np.array([[1.0, 0.0]])
TS = 0.05  # Sampling period (s).
Ad, Bd, Cd, _, _ = cont2discrete((A, B, C, np.zeros((1, 1))), TS, method="zoh")

Q = np.eye(2)
R = np.array([[0.1]])
X_INITIAL = np.zeros(2)
REFERENCE = 1.0  # Desired angular speed (rad/s).
DURATION = 3.0  # Simulation duration (s).
U_MIN, U_MAX = 0.0, 20.0


def steady_state_target(reference):
    """Solve x_s = Ad x_s + Bd u_s and Cd x_s = reference."""
    target_matrix = np.block([
        [np.eye(2) - Ad, -Bd],
        [Cd, np.zeros((1, 1))],
    ])
    target = np.linalg.solve(target_matrix, np.array([0.0, 0.0, reference]))
    return target[:2], float(target[2])


def make_mpc_controller(x_s, u_s, horizon, terminal_weight, constrained=True):
    """Build one parameterized QP and return a state-feedback controller.

    Optimization variables are deviations: delta_x = x - x_s and
    delta_u = u - u_s. The target cancels from the dynamics. Input bounds
    apply to the actual voltage delta_u + u_s, not to delta_u alone.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1")

    delta_x_initial = cp.Parameter(2, name="delta_x_initial")
    delta_x = cp.Variable((2, horizon + 1), name="delta_x")
    delta_u = cp.Variable((1, horizon), name="delta_u")
    constraints = [delta_x[:, 0] == delta_x_initial]
    objective = cp.quad_form(delta_x[:, horizon], terminal_weight)
    for i in range(horizon):
        objective += cp.quad_form(delta_x[:, i], Q)
        objective += cp.quad_form(delta_u[:, i], R)
        constraints.append(
            delta_x[:, i + 1] == Ad @ delta_x[:, i] + Bd @ delta_u[:, i]
        )
        if constrained:
            constraints += [
                U_MIN <= delta_u[:, i] + u_s,
                delta_u[:, i] + u_s <= U_MAX,
            ]
    problem = cp.Problem(cp.Minimize(objective), constraints)

    def controller(x):
        delta_x_initial.value = np.asarray(x) - x_s
        problem.solve(solver=cp.OSQP, warm_start=True, verbose=False,
                      eps_abs=1e-8, eps_rel=1e-8, max_iter=10000)
        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(f"MPC solve failed: {problem.status}")
        # Apply only the first move; optimize again at the next sample.
        return float(delta_u.value[0, 0] + u_s)

    return controller


def motorsim(controller, x_initial, steps):
    """Simulate the sampled motor; histories contain actual states and inputs."""
    states = np.empty((steps + 1, 2))
    inputs = np.empty(steps)
    states[0] = x_initial
    for k in range(steps):
        inputs[k] = controller(states[k])
        states[k + 1] = Ad @ states[k] + Bd[:, 0] * inputs[k]
    return states, inputs


def plot_comparison(lqr_response, mpc_response, horizon, terminal_cost, constrained):
    """Compare speed and actual voltage using the Matlab two-panel layout."""
    fig, (ax_y, ax_u) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for label, (states, inputs) in [
        ("LQR (unconstrained)", lqr_response), ("MPC", mpc_response)
    ]:
        ax_y.plot(np.arange(len(states)) * TS, (states @ Cd.T)[:, 0],
                  label=label, linewidth=1.7)
        ax_u.step(np.arange(len(inputs) + 1) * TS, np.r_[inputs, inputs[-1]],
                  where="post", label=label, linewidth=1.5)
    ax_y.axhline(REFERENCE, color="black", linestyle=":", label="Reference")
    if constrained:
        ax_u.axhline(U_MIN, color="gray", linestyle=":", label="MPC voltage bounds")
        ax_u.axhline(U_MAX, color="gray", linestyle=":")
    bounds = "0 <= u <= 20 V" if constrained else "no input bounds"
    ax_y.set(title=f"DC motor tracking: N={horizon}, terminal cost={terminal_cost}, {bounds}",
             ylabel="Angular speed (rad/s)")
    ax_u.set(xlabel="Time (s)", ylabel="Voltage (V)")
    ax_u.set_xlim(0, (len(mpc_response[1])) * TS)
    for ax in (ax_y, ax_u):
        ax.grid(True, alpha=0.35)
        ax.legend()
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--horizon", type=int, default=4, help="MPC horizon N (default: 4)")
    parser.add_argument("--terminal-cost", choices=("lqr", "none"), default="lqr")
    parser.add_argument("--unconstrained", action="store_true",
                        help="disable MPC voltage bounds; LQR is always unconstrained")
    args = parser.parse_args()
    if args.horizon < 1:
        parser.error("--horizon must be at least 1")

    x_s, u_s = steady_state_target(REFERENCE)
    # With this P, unconstrained finite-horizon MPC reproduces the infinite-horizon LQR.
    P = solve_discrete_are(Ad, Bd, Q, R)
    K = np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)

    def lqr_controller(x):
        return u_s - float((K @ (x - x_s)).item())

    terminal_weight = P if args.terminal_cost == "lqr" else np.zeros((2, 2))
    mpc_controller = make_mpc_controller(
        x_s, u_s, args.horizon, terminal_weight, constrained=not args.unconstrained
    )
    steps = int(np.ceil(DURATION / TS))
    lqr_response = motorsim(lqr_controller, X_INITIAL, steps)
    mpc_response = motorsim(mpc_controller, X_INITIAL, steps)

    print(f"Steady-state target: x_s = {x_s}, u_s = {u_s:.6g} V")
    for name, (states, inputs) in [("LQR", lqr_response), ("MPC", mpc_response)]:
        print(f"{name}: final speed = {(Cd @ states[-1]).item():.6g} rad/s, "
              f"voltage range = [{inputs.min():.6g}, {inputs.max():.6g}] V")
    plot_comparison(lqr_response, mpc_response, args.horizon,
                    args.terminal_cost, constrained=not args.unconstrained)
    plt.show()


if __name__ == "__main__":
    main()
