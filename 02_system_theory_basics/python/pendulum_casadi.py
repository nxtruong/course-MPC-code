"""Pendulum simulation, linearization, and discretization with CasADi."""

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np


def as_array(value):
    """Convert a CasADi vector or matrix to a NumPy array."""
    return np.asarray(value, dtype=float).squeeze()


def make_integrator(name, x, ode, step, parameter=None):
    dae = {"x": x, "ode": ode}
    if parameter is not None:
        dae["p"] = parameter
    options = {"abstol": 1e-11, "reltol": 1e-9}
    return ca.integrator(name, "cvodes", dae, 0.0, step, options)


def simulate_integrator(integrator, x0, steps, inputs=None):
    """Repeatedly evaluate a CasADi one-step integrator."""
    trajectory = np.zeros((len(x0), steps + 1))
    trajectory[:, 0] = x0
    for k in range(steps):
        arguments = {"x0": trajectory[:, k]}
        if inputs is not None:
            arguments["p"] = float(inputs[k])
        result = integrator(**arguments)
        trajectory[:, k + 1] = as_array(result["xf"])
    return trajectory


def pulse_input(t):
    t = np.asarray(t)
    return np.where(t < 1.0, 0.1, np.where(t < 2.0, -0.1, 0.0))


def main():
    # Adjustable nominal parameters from the lecture.
    g = 10.0                    # gravitational acceleration [m/s^2]
    length = 1.0               # pendulum length [m]
    ts = 0.01                  # simulation sampling time [s]

    x = ca.SX.sym("x", 2)
    u = ca.SX.sym("u")
    f = ca.vertcat(x[1], -(g / length) * ca.sin(x[0]) + u)
    h = x[0]
    dynamics = ca.Function("pendulum_dynamics", [x, u], [f])

    # %% Nonlinear continuous-time pendulum
    nonlinear_step = make_integrator("nonlinear_step", x, f, ts, u)
    t_sim = np.arange(0.0, 10.0 + ts, ts)
    u_decay = 0.1 * np.exp(-0.3 * t_sim) * np.sin(2.0 * t_sim)
    x_nonlinear = simulate_integrator(
        nonlinear_step,
        np.array([np.deg2rad(15.0), 0.0]),
        t_sim.size - 1,
        u_decay[:-1],
    )

    fig, axes = plt.subplots(2, 1, sharex=True, num="Nonlinear pendulum")
    axes[0].plot(t_sim, u_decay, linewidth=1.2)
    axes[0].set_ylabel(r"$u$ [rad/s$^2$]")
    axes[0].set_title("Decaying sinusoidal input")
    axes[1].plot(t_sim, np.rad2deg(x_nonlinear[0]), linewidth=1.2)
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel(r"$\theta$ [deg]")
    axes[1].set_title("Nonlinear pendulum output")
    for axis in axes:
        axis.grid(True)
    fig.tight_layout()

    # %% Linearization using CasADi automatic differentiation
    a_symbolic = ca.jacobian(f, x)
    b_symbolic = ca.jacobian(f, u)
    c_symbolic = ca.jacobian(h, x)
    d_symbolic = ca.jacobian(h, u)
    linearization = ca.Function(
        "linearization",
        [x, u],
        [a_symbolic, b_symbolic, c_symbolic, d_symbolic],
    )

    def analytical_jacobians(xs):
        a = np.array(
            [[0.0, 1.0], [-(g / length) * np.cos(xs[0]), 0.0]]
        )
        b = np.array([[0.0], [1.0]])
        c = np.array([[1.0, 0.0]])
        d = np.zeros((1, 1))
        return a, b, c, d

    # Case 1: equilibrium at theta = pi/4.
    xs1 = np.array([np.pi / 4.0, 0.0])
    us1 = (g / length) * np.sin(np.pi / 4.0)
    ad_results1 = linearization(xs1, us1)
    ac1_ad, bc_ad, c_ad, d_ad = [
        np.asarray(result, dtype=float) for result in ad_results1
    ]
    ac1_exact, bc_exact, c_exact, d_exact = analytical_jacobians(xs1)

    # Case 2: equilibrium at the origin with feedback u = -x2.
    xs2 = np.zeros(2)
    k_feedback = np.array([[0.0, -1.0]])
    ac2_ad = np.asarray(linearization(xs2, 0.0)[0], dtype=float)
    acl2_ad = ac2_ad + bc_ad @ k_feedback
    acl2_exact = analytical_jacobians(xs2)[0] + bc_exact @ k_feedback

    comparisons = {
        "Case 1 A": (ac1_ad, ac1_exact),
        "B": (bc_ad, bc_exact),
        "C": (c_ad, c_exact),
        "D": (d_ad, d_exact),
        "Case 2 Acl": (acl2_ad, acl2_exact),
    }
    print("Autodiff and analytical linearizations")
    for name, (automatic, analytical) in comparisons.items():
        np.testing.assert_allclose(automatic, analytical, atol=1e-12)
        error = np.max(np.abs(automatic - analytical))
        print(f"{name}: max abs. error = {error:.3e}")
    print("\nCase 1 Ac (CasADi autodiff):\n", ac1_ad)
    print("\nCase 2 Acl (CasADi autodiff):\n", acl2_ad)

    # %% Closed-loop nonlinear and linearized models: Case 2
    x_cl = ca.SX.sym("x_cl", 2)
    nonlinear_cl_ode = ca.substitute(f, ca.vertcat(x, u), ca.vertcat(x_cl, -x_cl[1]))

    dx_cl = ca.SX.sym("dx_cl", 2)
    linear_cl_ode = ca.DM(acl2_ad) @ dx_cl
    nonlinear_cl_step = make_integrator(
        "nonlinear_cl_step", x_cl, nonlinear_cl_ode, ts
    )
    linear_cl_step = make_integrator("linear_cl_step", dx_cl, linear_cl_ode, ts)

    t_closed_loop = np.arange(0.0, 20.0 + ts, ts)
    x0_small = np.deg2rad([5.0, 2.0])
    x_nonlinear_cl = simulate_integrator(
        nonlinear_cl_step, x0_small, t_closed_loop.size - 1
    )
    delta_x_linear_cl = simulate_integrator(
        linear_cl_step, x0_small - xs2, t_closed_loop.size - 1
    )
    x_linear_cl = xs2[:, None] + delta_x_linear_cl

    fig, axes = plt.subplots(
        2, 1, sharex=True, num="Closed-loop nonlinear and linearized models"
    )
    axes[0].plot(t_closed_loop, np.rad2deg(x_nonlinear_cl[0]), label="Nonlinear")
    axes[0].plot(
        t_closed_loop, np.rad2deg(x_linear_cl[0]), "--", label="Linearized"
    )
    axes[0].set_ylabel(r"$x_1$ [deg]")
    axes[0].set_title(r"Case 2: $u=-x_2$")
    axes[1].plot(t_closed_loop, np.rad2deg(x_nonlinear_cl[1]), label="Nonlinear")
    axes[1].plot(
        t_closed_loop, np.rad2deg(x_linear_cl[1]), "--", label="Linearized"
    )
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel(r"$x_2$ [deg/s]")
    for axis in axes:
        axis.grid(True)
        axis.legend()
    fig.tight_layout()

    # %% Euler and exact zero-order-hold discretization
    # Try 0.01 and 0.1 s to see the effect of the sampling time.
    ts_discretization = 0.01

    nonlinear_zoh_step = make_integrator(
        "nonlinear_zoh_step", x, f, ts_discretization, u
    )
    euler_map = ca.Function(
        "euler_map", [x, u], [x + ts_discretization * dynamics(x, u)]
    )

    dx = ca.MX.sym("dx", 2)
    delta_u_symbol = ca.MX.sym("delta_u")
    linear_ode = ca.DM(ac1_ad) @ dx + ca.DM(bc_ad) * delta_u_symbol
    linear_zoh_step = make_integrator(
        "linear_zoh_step", dx, linear_ode, ts_discretization, delta_u_symbol
    )

    # Differentiate the integration map to obtain the exact discrete A and B.
    dx0 = ca.MX.sym("dx0", 2)
    du0 = ca.MX.sym("du0")
    dx_next = linear_zoh_step(x0=dx0, p=du0)["xf"]
    discrete_jacobians = ca.Function(
        "discrete_jacobians",
        [dx0, du0],
        [ca.jacobian(dx_next, dx0), ca.jacobian(dx_next, du0)],
    )
    ad1, bd1 = [
        np.asarray(value, dtype=float)
        for value in discrete_jacobians(np.zeros(2), 0.0)
    ]

    t_compare = np.arange(0.0, 4.0 + ts_discretization, ts_discretization)
    delta_u = pulse_input(t_compare)
    x_ct = simulate_integrator(
        nonlinear_zoh_step,
        xs1,
        t_compare.size - 1,
        us1 + delta_u[:-1],
    )
    delta_x_linear = simulate_integrator(
        linear_zoh_step,
        np.zeros(2),
        t_compare.size - 1,
        delta_u[:-1],
    )

    x_euler = np.zeros_like(x_ct)
    x_euler[:, 0] = xs1
    for k in range(t_compare.size - 1):
        x_euler[:, k + 1] = as_array(
            euler_map(x_euler[:, k], us1 + delta_u[k])
        )
    x_linear = xs1[:, None] + delta_x_linear

    print("\nExact ZOH matrices from the CasADi integration map")
    print("Ad =\n", ad1)
    print("Bd =\n", bd1)

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
