"""DC motor load rejection: nominal MPC versus offset-free tracking MPC.

Run with ``python tracking_mpc_disturbance.py`` after installing the packages
in requirements.txt. Both controllers use N=4, a Riccati terminal cost, and
0 <= u <= 20 V. Only angular speed is measured; current is estimated.

At t=1 s, an unknown opposing load d=tau_L/J steps from 0 to 2 rad/s^2.
The nominal MPC uses a two-state Luenberger observer and a fixed target.
The offset-free MPC uses a three-state observer for [omega, i, d] and
recomputes its target from the estimated load at each sample. Observer
gains are chosen by pole placement. The simulation is noise-free.

The figure shows speed, applied voltage, and actual/estimated load. It is
displayed interactively and is not saved automatically. Edit the constants
below to change the experiment.
"""

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.signal import cont2discrete, place_poles


# x = [angular speed (rad/s), armature current (A)], u = voltage (V).
# The normalized load d = tau_L/J opposes rotation: omega_dot = -10*omega+i-d.
A = np.array([[-10.0, 1.0], [-0.02, -2.0]])
B = np.array([[0.0], [2.0]])
C = np.array([[1.0, 0.0]])
E = np.array([[-1.0], [0.0]])
TS = 0.05

# Discretize voltage and load channels together using an exact zero-order hold.
# x(k+1) = Ad*x(k) + Bd*u(k) + Ed*d(k), y(k) = Cd*x(k).
Ad, combined_input, Cd, _, _ = cont2discrete(
    (A, np.hstack((B, E)), C, np.zeros((1, 2))), TS, method="zoh"
)
Bd = combined_input[:, :1]
Ed = combined_input[:, 1:]

Q = np.eye(2)
R = np.array([[0.1]])
P = solve_discrete_are(Ad, Bd, Q, R)
HORIZON = 4
U_MIN, U_MAX = 0.0, 20.0
REFERENCE = 1.0
X_INITIAL = np.zeros(2)
DURATION = 5.0
LOAD_TIME = 1.0
LOAD_MAGNITUDE = 2.0

# Eigenvalues of A_observer - L*C_observer (discrete-time predictor form).
NOMINAL_OBSERVER_POLES = np.array([0.4, 0.5])
AUGMENTED_OBSERVER_POLES = np.array([0.4, 0.5, 0.6])


def steady_state_target(reference, disturbance_estimate):
    """Find an equilibrium with the estimated load and the requested speed.

    (I-Ad)*x_s - Bd*u_s = Ed*d_hat, Cd*x_s = reference.
    The actual plant load is never passed to this function by the controller.
    """
    target_matrix = np.block([
        [np.eye(2) - Ad, -Bd],
        [Cd, np.zeros((1, 1))],
    ])
    rhs = np.r_[Ed[:, 0] * disturbance_estimate, reference]
    target = np.linalg.solve(target_matrix, rhs)
    x_s, u_s = target[:2], float(target[2])
    if not U_MIN <= u_s <= U_MAX:
        raise ValueError(
            f"Target voltage {u_s:.6g} V is outside [{U_MIN}, {U_MAX}] V; "
            "the requested reference/load combination is not feasible."
        )
    return x_s, u_s


def make_observer(offset_free):
    """Return predictor-form observer matrices, using speed as the only output."""
    if offset_free:
        # d(k+1)=d(k) stores the unknown constant load; Cd_disturbance=0.
        A_observer = np.block([[Ad, Ed], [np.zeros((1, 2)), np.ones((1, 1))]])
        B_observer = np.vstack((Bd, np.zeros((1, 1))))
        C_observer = np.hstack((Cd, np.zeros((1, 1))))
        poles = AUGMENTED_OBSERVER_POLES
    else:
        A_observer, B_observer, C_observer = Ad, Bd, Cd
        poles = NOMINAL_OBSERVER_POLES
    L = place_poles(A_observer.T, C_observer.T, poles).gain_matrix.T
    return A_observer, B_observer, C_observer, L


def make_mpc_controller():
    """Build the common deviation-coordinate MPC once, with adjustable targets.

    With d_i=d_hat over the horizon and a disturbance-consistent target,
    the load cancels: delta_x(i+1)=Ad*delta_x(i)+Bd*delta_u(i).
    The nominal MPC uses the same problem with d_hat=0 and a fixed target.
    """
    delta_x_initial = cp.Parameter(2, name="delta_x_initial")
    target_voltage = cp.Parameter(name="u_s")
    delta_x = cp.Variable((2, HORIZON + 1), name="delta_x")
    delta_u = cp.Variable((1, HORIZON), name="delta_u")
    constraints = [delta_x[:, 0] == delta_x_initial]
    objective = cp.quad_form(delta_x[:, HORIZON], P)
    for i in range(HORIZON):
        objective += cp.quad_form(delta_x[:, i], Q)
        objective += cp.quad_form(delta_u[:, i], R)
        constraints += [
            delta_x[:, i + 1] == Ad @ delta_x[:, i] + Bd @ delta_u[:, i],
            U_MIN <= delta_u[:, i] + target_voltage,
            delta_u[:, i] + target_voltage <= U_MAX,
        ]
    problem = cp.Problem(cp.Minimize(objective), constraints)

    def controller(state_estimate, x_s, u_s):
        delta_x_initial.value = state_estimate - x_s
        target_voltage.value = u_s
        problem.solve(solver=cp.OSQP, warm_start=True, verbose=False,
                      eps_abs=1e-8, eps_rel=1e-8, max_iter=10000)
        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(f"MPC solve failed: {problem.status}")
        return float(delta_u.value[0, 0] + u_s)

    return controller


def simulate(offset_free, load_magnitude=LOAD_MAGNITUDE):
    """Simulate one closed loop; expose true states/load only to the plant.

    At sample k, the controller uses the prior estimate x_hat(k). The
    innovation y_m(k)-C_observer*z_hat(k) and applied u(k) predict z_hat(k+1).
    Thus measurement and control timing matches the observer in the slides.
    """
    A_observer, B_observer, C_observer, L = make_observer(offset_free)
    controller = make_mpc_controller()
    steps = int(np.ceil(DURATION / TS))
    # Roundoff-safe step index; a non-grid-aligned LOAD_TIME takes effect at
    # the first sample at or after that time.
    load_step = int(np.ceil(LOAD_TIME / TS - 1e-12))
    states = np.empty((steps + 1, 2))
    estimates = np.zeros((steps + 1, A_observer.shape[0]))
    inputs = np.empty(steps)
    disturbance = np.zeros(steps)
    disturbance[max(0, load_step):] = load_magnitude
    state_targets = np.empty((steps, 2))
    input_targets = np.empty(steps)
    innovations = np.empty(steps)
    states[0] = X_INITIAL
    # Both observers start from zero and receive no true current or load.
    nominal_target = steady_state_target(REFERENCE, 0.0)

    for k in range(steps):
        y_measured = (Cd @ states[k]).item()
        innovation = y_measured - (C_observer @ estimates[k]).item()
        if offset_free:
            x_s, u_s = steady_state_target(REFERENCE, estimates[k, 2])
        else:
            x_s, u_s = nominal_target
        inputs[k] = controller(estimates[k, :2], x_s, u_s)

        # Predictor observer: z_hat(k+1)=A_obs*z_hat(k)+B_obs*u(k)+L*e(k).
        estimates[k + 1] = (A_observer @ estimates[k]
                            + B_observer[:, 0] * inputs[k]
                            + L[:, 0] * innovation)
        states[k + 1] = (Ad @ states[k] + Bd[:, 0] * inputs[k]
                         + Ed[:, 0] * disturbance[k])
        state_targets[k], input_targets[k] = x_s, u_s
        innovations[k] = innovation

    return {
        "states": states, "estimates": estimates, "inputs": inputs,
        "disturbance": disturbance, "state_targets": state_targets,
        "input_targets": input_targets, "innovations": innovations,
    }


def plot_comparison(nominal, offset_free):
    """Show the speed offset, extra load-compensating voltage, and load estimate."""
    fig, (ax_y, ax_u, ax_d) = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    time = np.arange(len(offset_free["states"])) * TS
    for label, response, color in [
        ("Nominal MPC", nominal, "tab:blue"),
        ("Offset-free MPC", offset_free, "tab:orange"),
    ]:
        ax_y.plot(time, (response["states"] @ Cd.T)[:, 0],
                  label=label, color=color, linewidth=1.8)
        ax_u.step(time, np.r_[response["inputs"], response["inputs"][-1]],
                  where="post", label=label, color=color, linewidth=1.5)
    ax_y.axhline(REFERENCE, color="black", linestyle=":", label="Reference")
    ax_y.set(title=f"Load rejection: nominal and offset-free MPC (N={HORIZON}, Riccati terminal cost)",
             ylabel="Angular speed (rad/s)")
    ax_u.axhline(U_MIN, color="gray", linestyle=":", label="Voltage bounds")
    ax_u.axhline(U_MAX, color="gray", linestyle=":")
    ax_u.set_ylabel("Voltage (V)")
    ax_d.step(time, np.r_[offset_free["disturbance"], offset_free["disturbance"][-1]],
              where="post", color="black", linestyle="--", label="Actual load")
    ax_d.plot(time, offset_free["estimates"][:, 2], color="tab:orange",
              label="Estimated load (offset-free MPC)", linewidth=1.8)
    ax_d.set(xlabel="Time (s)", ylabel=r"Load $d=\tau_L/J$ (rad/s$^2$)")
    for ax in (ax_y, ax_u, ax_d):
        ax.axvline(LOAD_TIME, color="gray", linestyle="--", alpha=0.6)
        ax.grid(True, alpha=0.35)
        ax.legend()
    ax_d.set_xlim(time[0], time[-1])
    fig.tight_layout()
    return fig


def main():
    nominal = simulate(offset_free=False)
    offset_free = simulate(offset_free=True)
    print(f"Unknown load: {LOAD_MAGNITUDE:g} rad/s^2 applied at t={LOAD_TIME:g} s")
    print(f"Both MPCs: N={HORIZON}, Riccati terminal cost, {U_MIN:g} <= u <= {U_MAX:g} V")
    for label, response in [("Nominal MPC", nominal), ("Offset-free MPC", offset_free)]:
        speed = (Cd @ response["states"][-1]).item()
        voltage = response["inputs"]
        print(f"{label}: final speed={speed:.6f} rad/s, error r-y={REFERENCE-speed:.6f}, "
              f"voltage range=[{voltage.min():.6f}, {voltage.max():.6f}] V")
    print(f"Final estimated load: {offset_free['estimates'][-1, 2]:.6f} rad/s^2")
    print(f"Final offset-free target: x_s={offset_free['state_targets'][-1]}, "
          f"u_s={offset_free['input_targets'][-1]:.6f} V")
    plot_comparison(nominal, offset_free)
    plt.show()


if __name__ == "__main__":
    main()
