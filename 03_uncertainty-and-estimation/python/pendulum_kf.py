"""Kalman filter for a pendulum linearized about a nonzero equilibrium."""

import control as ct
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


def pendulum_dynamics(x, u, g, length):
    """Return xdot for normalized torque u = T_c/(m*l^2)."""
    return np.array([x[1], -(g / length) * np.sin(x[0]) + u])


def pulse_input(t):
    """Return the requested input deviation about the equilibrium input."""
    t = np.asarray(t)
    return np.where(t < 1.0, 0.1, np.where(t < 2.0, -0.1, 0.0))


def main():
    rng = np.random.default_rng(4)

    # Nonlinear continuous-time pendulum parameters.
    g = 10.0                    # gravitational acceleration [m/s^2]
    length = 1.0               # pendulum length [m]
    ts = 0.01                  # adjustable sampling time [s]

    # Linearize about (xs, us) and discretize exactly under zero-order hold.
    xs = np.array([np.pi / 4.0, 0.0])
    us = (g / length) * np.sin(xs[0])
    ac = np.array(
        [[0.0, 1.0], [-(g / length) * np.cos(xs[0]), 0.0]]
    )
    bc = np.array([[0.0], [1.0]])
    c = np.array([[1.0, 0.0]])       # only the angle is measured
    d = np.zeros((1, 1))

    system_c = ct.ss(ac, bc, c, d)
    system_d = ct.sample_system(system_c, ts, method="zoh")
    a = np.asarray(system_d.A)
    b = np.asarray(system_d.B)

    # Known discrete-time Gaussian disturbance covariances.
    r1 = np.diag(np.deg2rad([0.02, 0.05]) ** 2)
    r2 = np.deg2rad(1.0) ** 2
    l1 = np.linalg.cholesky(r1)

    x_hat0 = xs + np.deg2rad([5.0, -3.0])
    p0 = np.diag(np.deg2rad([5.0, 5.0]) ** 2)

    t_final = 5.0
    n_steps = round(t_final / ts)
    time = ts * np.arange(n_steps + 1)
    delta_u = pulse_input(time)

    x = np.zeros((2, time.size))
    y = np.zeros(time.size)
    x_hat = np.zeros_like(x)
    x[:, 0] = xs

    # Correct the initial prior estimate using the first measurement.
    y[0] = (c @ x[:, 0]).item() + np.sqrt(r2) * rng.standard_normal()
    innovation_cov = (c @ p0 @ c.T).item() + r2
    kalman_gain = p0 @ c.T / innovation_cov
    innovation = y[0] - (c @ x_hat0).item()
    x_hat[:, 0] = x_hat0 + kalman_gain[:, 0] * innovation
    identity_minus_kc = np.eye(2) - kalman_gain @ c
    p = (
        identity_minus_kc @ p0 @ identity_minus_kc.T
        + r2 * kalman_gain @ kalman_gain.T
    )

    for k in range(time.size - 1):
        # Simulate one nonlinear zero-order-hold interval.
        u_total = us + delta_u[k]
        interval = solve_ivp(
            lambda _t, state: pendulum_dynamics(state, u_total, g, length),
            (time[k], time[k + 1]),
            x[:, k],
            rtol=1e-9,
            atol=1e-11,
        )
        x[:, k + 1] = interval.y[:, -1] + l1 @ rng.standard_normal(2)
        y[k + 1] = (
            (c @ x[:, k + 1]).item() + np.sqrt(r2) * rng.standard_normal()
        )

        # Predict in deviation variables, then return to absolute state units.
        delta_x_pred = a @ (x_hat[:, k] - xs) + b[:, 0] * delta_u[k]
        x_pred = xs + delta_x_pred
        p_pred = a @ p @ a.T + r1

        # Measurement correction and Joseph covariance update.
        innovation_cov = (c @ p_pred @ c.T).item() + r2
        kalman_gain = p_pred @ c.T / innovation_cov
        innovation = y[k + 1] - (c @ x_pred).item()
        x_hat[:, k + 1] = x_pred + kalman_gain[:, 0] * innovation
        identity_minus_kc = np.eye(2) - kalman_gain @ c
        p = (
            identity_minus_kc @ p_pred @ identity_minus_kc.T
            + r2 * kalman_gain @ kalman_gain.T
        )
        p = (p + p.T) / 2.0

    fig, axes = plt.subplots(
        3, 1, sharex=True, num="Pendulum Kalman filter"
    )
    axes[0].plot(time, np.rad2deg(x[0]), "k", linewidth=1.2, label="Actual")
    axes[0].plot(
        time, np.rad2deg(x_hat[0]), "--", linewidth=1.2, label="Kalman estimate"
    )
    axes[0].plot(
        time,
        np.rad2deg(y),
        ".",
        color=(0.65, 0.75, 0.90),
        markersize=3,
        label="Measurement",
    )
    axes[0].set_ylabel(r"$\theta$ [deg]")
    axes[0].set_title("Nonlinear pendulum with an LTI Kalman filter")

    axes[1].plot(time, np.rad2deg(x[1]), "k", linewidth=1.2, label="Actual")
    axes[1].plot(
        time, np.rad2deg(x_hat[1]), "--", linewidth=1.2, label="Kalman estimate"
    )
    axes[1].set_ylabel(r"$\omega$ [deg/s]")

    axes[2].step(time, delta_u, where="post", linewidth=1.2)
    axes[2].set_xlabel("Time [s]")
    axes[2].set_ylabel(r"$\Delta u$ [rad/s$^2$]")
    axes[2].set_ylim(-0.12, 0.12)

    for axis in axes:
        axis.grid(True)
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()

    rmse = np.sqrt(np.mean((x_hat - x) ** 2, axis=1))
    print(
        "State-estimation RMSE: "
        f"{np.rad2deg(rmse[0]):.3f} deg, "
        f"{np.rad2deg(rmse[1]):.3f} deg/s"
    )
    plt.show()


if __name__ == "__main__":
    main()
