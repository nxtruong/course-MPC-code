"""Nonlinear pendulum, linearization, and discretization examples."""

import control as ct
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


def pendulum_dynamics(x, u, g, length):
    """Return xdot for normalized torque u = T_c/(m*l^2)."""
    return np.array([x[1], -(g / length) * np.sin(x[0]) + u])


def pendulum_update(_t, x, u, params):
    return pendulum_dynamics(x, np.atleast_1d(u)[0], params["g"], params["length"])


def pendulum_output(_t, x, _u, _params):
    return np.array([x[0]])


def pulse_input(t):
    """Input used for the discretization comparison."""
    t = np.asarray(t)
    return np.where(t < 1.0, 0.1, np.where(t < 2.0, -0.1, 0.0))


def main():
    # Adjustable nominal parameters from the lecture.
    g = 10.0                    # gravitational acceleration [m/s^2]
    length = 1.0               # pendulum length [m]
    ts = 0.01                  # sampling time [s]

    pendulum = ct.nlsys(
        pendulum_update,
        pendulum_output,
        inputs=["u"],
        outputs=["theta"],
        states=["theta", "theta_dot"],
        params={"g": g, "length": length},
    )

    # Nonlinear continuous-time response to a small decaying sine input.
    t_sim = np.arange(0.0, 10.0 + ts, ts)
    u_decay = 0.1 * np.exp(-0.3 * t_sim) * np.sin(2.0 * t_sim)
    response = ct.input_output_response(
        pendulum,
        t_sim,
        u_decay,
        initial_state=[np.deg2rad(15.0), 0.0],
        return_states=True,
        solve_ivp_kwargs={"rtol": 1e-9, "atol": 1e-11},
    )

    fig, axes = plt.subplots(2, 1, sharex=True, num="Nonlinear pendulum")
    axes[0].plot(response.time, u_decay, linewidth=1.2)
    axes[0].set_ylabel(r"$u$ [rad/s$^2$]")
    axes[0].set_title("Decaying sinusoidal input")
    axes[1].plot(response.time, np.rad2deg(response.outputs), linewidth=1.2)
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel(r"$\theta$ [deg]")
    axes[1].set_title("Nonlinear pendulum output")
    for axis in axes:
        axis.grid(True)
    fig.tight_layout()

    # General deviation model about an equilibrium (xs, us):
    #   Delta xdot = Ac(xs) Delta x + Bc Delta u
    #   Delta y    = C Delta x + D Delta u
    def a_continuous(xs):
        return np.array([[0.0, 1.0], [-(g / length) * np.cos(xs[0]), 0.0]])

    bc = np.array([[0.0], [1.0]])
    c = np.array([[1.0, 0.0]])
    d = np.zeros((1, 1))

    # Case 1: equilibrium at theta = pi/4.
    xs1 = np.array([np.pi / 4.0, 0.0])
    us1 = (g / length) * np.sin(np.pi / 4.0)
    ac1 = a_continuous(xs1)

    # Case 2: equilibrium at the origin with feedback u = -x2.
    xs2 = np.zeros(2)
    k_feedback = np.array([[0.0, -1.0]])
    acl2 = a_continuous(xs2) + bc @ k_feedback

    t_closed_loop = np.arange(0.0, 20.0 + ts, ts)
    x0_small = np.deg2rad([5.0, 2.0])
    nonlinear_cl = solve_ivp(
        lambda _t, x: pendulum_dynamics(x, -x[1], g, length),
        (t_closed_loop[0], t_closed_loop[-1]),
        x0_small,
        t_eval=t_closed_loop,
        rtol=1e-9,
        atol=1e-11,
    )
    linear_cl = solve_ivp(
        lambda _t, dx: acl2 @ dx,
        (t_closed_loop[0], t_closed_loop[-1]),
        x0_small - xs2,
        t_eval=t_closed_loop,
        rtol=1e-9,
        atol=1e-11,
    )
    x_linear_cl = xs2[:, None] + linear_cl.y

    fig, axes = plt.subplots(
        2, 1, sharex=True, num="Closed-loop nonlinear and linearized models"
    )
    axes[0].plot(t_closed_loop, np.rad2deg(nonlinear_cl.y[0]), label="Nonlinear")
    axes[0].plot(
        t_closed_loop, np.rad2deg(x_linear_cl[0]), "--", label="Linearized"
    )
    axes[0].set_ylabel(r"$x_1$ [deg]")
    axes[0].set_title(r"Case 2: $u=-x_2$")
    axes[1].plot(t_closed_loop, np.rad2deg(nonlinear_cl.y[1]), label="Nonlinear")
    axes[1].plot(
        t_closed_loop, np.rad2deg(x_linear_cl[1]), "--", label="Linearized"
    )
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel(r"$x_2$ [deg/s]")
    for axis in axes:
        axis.grid(True)
        axis.legend()
    fig.tight_layout()

    # Exact zero-order-hold discretization of the Case 1 linear model.

    # sampling time for discretization [s]
    # Try small (0.01) and large (0.1) to see the effect
    ts = 0.01

    system_c1 = ct.ss(ac1, bc, c, d)
    system_d1 = ct.sample_system(system_c1, ts, method="zoh")
    ad1, bd1 = np.asarray(system_d1.A), np.asarray(system_d1.B)

    t_compare = np.arange(0.0, 4.0 + ts, ts)
    delta_u = pulse_input(t_compare)       # deviation from us1
    x_ct = np.zeros((2, t_compare.size))
    x_euler = np.zeros_like(x_ct)
    delta_x_linear = np.zeros_like(x_ct)
    x_ct[:, 0] = xs1
    x_euler[:, 0] = xs1

    for k in range(t_compare.size - 1):
        u_total = us1 + delta_u[k]

        # Integrate each zero-order-hold interval separately.
        interval = solve_ivp(
            lambda _t, x: pendulum_dynamics(x, u_total, g, length),
            (t_compare[k], t_compare[k + 1]),
            x_ct[:, k],
            rtol=1e-9,
            atol=1e-11,
        )
        x_ct[:, k + 1] = interval.y[:, -1]

        # Nonlinear forward-Euler model.
        x_euler[:, k + 1] = x_euler[:, k] + ts * pendulum_dynamics(
            x_euler[:, k], u_total, g, length
        )
        delta_x_linear[:, k + 1] = (
            ad1 @ delta_x_linear[:, k] + bd1[:, 0] * delta_u[k]
        )

    x_linear = xs1[:, None] + delta_x_linear

    fig, axes = plt.subplots(
        3, 1, sharex=True, num="Continuous-time and discrete-time comparison"
    )
    labels = ("Nonlinear CT", "Nonlinear Euler", "Linear exact ZOH")
    styles = ("-", "--", ":")
    trajectories = (x_ct, x_euler, x_linear)
    for trajectory, style, label in zip(trajectories, styles, labels):
        axes[0].plot(t_compare, np.rad2deg(trajectory[0]), style, label=label)
        axes[1].plot(t_compare, np.rad2deg(trajectory[1]), style, label=label)
    axes[0].set_ylabel(r"$x_1$ [deg]")
    axes[0].set_title(r"Case 1 models about $x_s=[\pi/4,0]^T$")
    axes[1].set_ylabel(r"$x_2$ [deg/s]")
    axes[2].step(t_compare, delta_u, where="post")
    axes[2].set_xlabel("Time [s]")
    axes[2].set_ylabel(r"$\Delta u$ [rad/s$^2$]")
    for axis in axes:
        axis.grid(True)
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
