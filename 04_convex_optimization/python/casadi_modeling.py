"""CasADi QP and NLP modeling examples for Lecture 4."""

from __future__ import annotations

import casadi as ca
import numpy as np


def _quiet_ipopt(opti: ca.Opti):
    """Select IPOPT while suppressing its normal iteration transcript."""
    opti.solver(
        "ipopt",
        {"print_time": False},
        {"print_level": 0, "sb": "yes"},
    )


def _require_plugin(kind: str, name: str):
    """Give a focused setup error when a bundled CasADi plugin is absent."""
    checker = ca.has_conic if kind == "conic" else ca.has_nlpsol
    if not checker(name):
        raise RuntimeError(
            f"CasADi's {name} {kind} plugin is unavailable. Install a CasADi "
            f"wheel that includes {name}."
        )


def box_qp_example(*, display=True):
    """Solve the lecture box-projection QP with CasADi Opti and qpOASES."""
    opti = ca.Opti("conic")
    x = opti.variable(2)
    target = ca.DM([3.0, 2.0])
    objective = ca.sumsqr(x - target)
    opti.minimize(objective)

    # Explicit, lecture-ordered constraints make the dual ordering transparent.
    constraints = [
        x[0] - 1 <= 0,
        -x[0] - 1 <= 0,
        x[1] - 1 <= 0,
        -x[1] - 1 <= 0,
    ]
    for constraint in constraints:
        opti.subject_to(constraint)

    _require_plugin("conic", "qpoases")
    opti.solver("qpoases", {"printLevel": "none"})
    try:
        solution = opti.solve()
    except RuntimeError as error:
        raise RuntimeError(
            "CasADi could not use its qpOASES plugin. Install a CasADi wheel "
            "that includes qpOASES."
        ) from error

    x_star = np.asarray(solution.value(x), dtype=float).reshape(-1)
    dual = np.array(
        [float(solution.value(opti.dual(constraint))) for constraint in constraints]
    )
    objective_value = float(solution.value(objective))

    constraint_matrix = np.array(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
    )
    g = constraint_matrix @ x_star - 1.0
    stationarity = 2.0 * (x_star - np.array([3.0, 2.0])) + constraint_matrix.T @ dual
    metrics = {
        "primal_violation": float(np.linalg.norm(np.maximum(g, 0.0), np.inf)),
        "dual_violation": float(np.linalg.norm(np.maximum(-dual, 0.0), np.inf)),
        "complementarity": float(np.linalg.norm(dual * g, np.inf)),
        "stationarity": float(np.linalg.norm(stationarity, np.inf)),
    }

    np.testing.assert_allclose(x_star, [1.0, 1.0], atol=2e-6)
    np.testing.assert_allclose(objective_value, 5.0, atol=2e-6)
    np.testing.assert_allclose(dual, [4.0, 0.0, 2.0, 0.0], atol=2e-5)
    for name, value in metrics.items():
        assert value < 2e-5, (name, value)

    if display:
        stats = solution.stats()
        print("=== CasADi Opti: box-projection QP ===")
        print(f"return status:       {stats.get('return_status', 'not reported')}")
        print(f"x*:                  {x_star}")
        print(f"objective:           {objective_value:.9g}")
        print(f"dual:                {dual}")
        print(f"iterations:          {stats.get('iter_count', 'not reported')}")
        print("KKT residuals:")
        for name, value in metrics.items():
            print(f"  {name:20s} {value:.3e}")

    return opti, solution, metrics


def nonlinear_projection_example(*, display=True):
    """Project [1, 2] onto the Euclidean unit ball with CasADi and IPOPT."""
    opti = ca.Opti()
    x = opti.variable(2)
    target = ca.DM([1.0, 2.0])
    objective = ca.sumsqr(x - target)
    ball_constraint = ca.sumsqr(x) - 1 <= 0
    opti.minimize(objective)
    opti.subject_to(ball_constraint)
    opti.set_initial(x, [0.0, 0.0])  # A strictly feasible point (Slater point).
    _require_plugin("nlpsol", "ipopt")
    _quiet_ipopt(opti)

    try:
        solution = opti.solve()
    except RuntimeError as error:
        raise RuntimeError(
            "CasADi could not use its IPOPT plugin. Install a CasADi wheel "
            "that includes IPOPT."
        ) from error

    x_star = np.asarray(solution.value(x), dtype=float).reshape(-1)
    dual = float(solution.value(opti.dual(ball_constraint)))
    objective_value = float(solution.value(objective))

    # Ask CasADi, rather than hand coding, for the Lagrangian gradient.
    x_symbol = ca.MX.sym("x", 2)
    dual_symbol = ca.MX.sym("lambda")
    symbolic_objective = ca.sumsqr(x_symbol - target)
    symbolic_constraint = ca.sumsqr(x_symbol) - 1
    lagrangian_gradient = ca.Function(
        "lagrangian_gradient",
        [x_symbol, dual_symbol],
        [ca.gradient(symbolic_objective + dual_symbol * symbolic_constraint, x_symbol)],
    )
    stationarity = np.asarray(lagrangian_gradient(x_star, dual), dtype=float).reshape(-1)
    g_value = float(x_star @ x_star - 1.0)
    metrics = {
        "primal_violation": max(g_value, 0.0),
        "dual_violation": max(-dual, 0.0),
        "complementarity": abs(dual * g_value),
        "stationarity": float(np.linalg.norm(stationarity, np.inf)),
    }

    expected_x = np.array([1.0, 2.0]) / np.sqrt(5.0)
    expected_objective = (np.sqrt(5.0) - 1.0) ** 2
    expected_dual = np.sqrt(5.0) - 1.0
    np.testing.assert_allclose(x_star, expected_x, atol=2e-6)
    np.testing.assert_allclose(objective_value, expected_objective, atol=2e-6)
    np.testing.assert_allclose(dual, expected_dual, atol=2e-6)
    for name, value in metrics.items():
        assert value < 2e-6, (name, value)

    if display:
        stats = solution.stats()
        print("\n=== CasADi Opti: nonlinear projection onto the unit ball ===")
        print(f"return status:       {stats.get('return_status', 'not reported')}")
        print(f"x*:                  {x_star}")
        print(f"objective:           {objective_value:.9g}")
        print(f"constraint dual:     {dual:.9g}")
        print(f"iterations:          {stats.get('iter_count', 'not reported')}")
        print("KKT residuals (computed with CasADi derivatives):")
        for name, value in metrics.items():
            print(f"  {name:20s} {value:.3e}")
        print(
            "Because the problem is convex and x=0 is strictly feasible, "
            "these KKT checks certify global optimality up to numerical tolerance."
        )

    return opti, solution, metrics


def infeasible_opti_example(*, display=True):
    """Catch an infeasible IPOPT solve and use Opti's debug diagnostics."""
    opti = ca.Opti()
    x = opti.variable()
    opti.minimize(x**2)
    opti.subject_to(x <= -1)
    opti.subject_to(x >= 1)
    opti.set_initial(x, 0.0)
    _require_plugin("nlpsol", "ipopt")
    _quiet_ipopt(opti)

    caught_error = None
    try:
        opti.solve()
    except RuntimeError as error:
        caught_error = error

    assert caught_error is not None, "The deliberately infeasible problem unexpectedly solved."
    stats = opti.stats()
    status = stats.get("return_status", "not reported")

    if display:
        print("\n=== CasADi Opti: infeasible problem diagnostics ===")
        print(f"solve raised RuntimeError; return status: {status}")
        print("Constraints reported by opti.debug.show_infeasibilities():")
        opti.debug.show_infeasibilities(1e-7)
        print(
            "Debug iterates help locate conflicting constraints; they are not "
            "a feasible solution or a formal infeasibility certificate."
        )

    return opti, status


def main():
    np.set_printoptions(precision=6, suppress=True)
    box_qp_example()
    nonlinear_projection_example()
    infeasible_opti_example()


if __name__ == "__main__":
    main()
