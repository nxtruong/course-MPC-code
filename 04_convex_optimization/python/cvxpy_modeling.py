"""CVXPY modeling examples for Lecture 4: Convex Optimization."""

from __future__ import annotations

import cvxpy as cp
import numpy as np


OPTIMAL_STATUSES = {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}


def require_solution(problem: cp.Problem):
    """Reject failed solves before reading variable or dual values."""
    if problem.status not in OPTIMAL_STATUSES:
        raise RuntimeError(f"CVXPY did not solve the problem: {problem.status}")


def production_lp_example(*, display=True):
    """Solve a production LP and verify its primal-dual certificate."""
    profit = np.array([3.0, 2.0])
    resource_matrix = np.array(
        [
            [1.0, 1.0],
            [2.0, 1.0],
            [1.0, 0.0],
        ]
    )
    resource_limits = cp.Parameter(3, value=np.array([4.0, 5.0, 3.0]))
    x = cp.Variable(2, name="production")
    resource_constraint = resource_matrix @ x <= resource_limits
    nonnegative_constraint = x >= 0
    problem = cp.Problem(
        cp.Maximize(profit @ x),
        [resource_constraint, nonnegative_constraint],
    )

    if cp.CLARABEL not in cp.installed_solvers():
        raise RuntimeError("CLARABEL is required; reinstall CVXPY with its defaults.")
    problem.solve(solver=cp.CLARABEL, verbose=False)
    require_solution(problem)

    x_star = np.asarray(x.value)
    resource_dual = np.asarray(resource_constraint.dual_value)
    lower_dual = np.asarray(nonnegative_constraint.dual_value)
    limits = np.asarray(resource_limits.value)
    slacks = limits - resource_matrix @ x_star
    dual_slacks = resource_matrix.T @ resource_dual - profit
    dual_objective = float(limits @ resource_dual)
    metrics = {
        "primal_violation": float(np.linalg.norm(np.minimum(slacks, 0.0), np.inf)),
        "dual_violation": float(
            max(
                np.linalg.norm(np.minimum(resource_dual, 0.0), np.inf),
                np.linalg.norm(np.minimum(dual_slacks, 0.0), np.inf),
            )
        ),
        "resource_complementarity": float(
            np.linalg.norm(resource_dual * slacks, np.inf)
        ),
        "bound_complementarity": float(np.linalg.norm(lower_dual * x_star, np.inf)),
        "stationarity": float(
            np.linalg.norm(resource_matrix.T @ resource_dual - lower_dual - profit, np.inf)
        ),
        "duality_gap": dual_objective - float(problem.value),
    }

    np.testing.assert_allclose(x_star, [1.0, 3.0], atol=2e-6)
    np.testing.assert_allclose(problem.value, 9.0, atol=2e-6)
    np.testing.assert_allclose(resource_dual, [1.0, 1.0, 0.0], atol=2e-6)
    for name, value in metrics.items():
        assert abs(value) < 2e-6, (name, value)

    base_value = float(problem.value)
    base_shadow_price = float(resource_dual[0])
    perturbation = 0.1
    predicted_value = base_value + base_shadow_price * perturbation
    resource_limits.value = np.array([4.0 + perturbation, 5.0, 3.0])
    problem.solve(solver=cp.CLARABEL, warm_start=True, verbose=False)
    require_solution(problem)
    perturbed_value = float(problem.value)
    np.testing.assert_allclose(predicted_value, 9.1, atol=2e-6)
    np.testing.assert_allclose(perturbed_value, predicted_value, atol=2e-6)

    if display:
        print("=== CVXPY production LP ===")
        print(f"is DCP:                        {problem.is_dcp()}")
        print(f"status:                        {problem.status}")
        print(f"x*:                            {x_star}")
        print(f"optimal profit:                {base_value:.9g}")
        print(f"resource slacks:               {slacks}")
        print(f"resource duals:                {resource_dual}")
        print(f"nonnegativity duals:           {lower_dual}")
        print(f"primal-dual gap:               {metrics['duality_gap']:.3e}")
        print(f"predicted perturbed profit:    {predicted_value:.9g}")
        print(f"resolved perturbed profit:     {perturbed_value:.9g}")

    resource_limits.value = np.array([4.0, 5.0, 3.0])
    return problem, metrics


def box_qp_kkt(target, x, dual):
    """Return KKT residuals for the four lecture-ordered inequalities."""
    target = np.asarray(target)
    x = np.asarray(x)
    dual = np.asarray(dual)
    constraint_matrix = np.array(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
    )
    g = constraint_matrix @ x - 1.0
    return {
        "primal_violation": float(np.linalg.norm(np.maximum(g, 0.0), np.inf)),
        "dual_violation": float(np.linalg.norm(np.maximum(-dual, 0.0), np.inf)),
        "complementarity": float(np.linalg.norm(dual * g, np.inf)),
        "stationarity": float(
            np.linalg.norm(2.0 * (x - target) + constraint_matrix.T @ dual, np.inf)
        ),
    }


def box_qp_example(*, display=True):
    """Model the lecture QP once, then solve it with interchangeable solvers."""
    target = cp.Parameter(2, value=np.array([3.0, 2.0]), name="target")
    x = cp.Variable(2, name="x")
    constraints = [
        x[0] <= 1,
        -x[0] <= 1,
        x[1] <= 1,
        -x[1] <= 1,
    ]
    problem = cp.Problem(cp.Minimize(cp.sum_squares(x - target)), constraints)
    assert problem.is_dcp()

    requested_solvers = [cp.OSQP, cp.CLARABEL]
    installed = set(cp.installed_solvers())
    available_solvers = [solver for solver in requested_solvers if solver in installed]
    if cp.OSQP not in available_solvers:
        raise RuntimeError("OSQP is required for this example.")

    solutions = {}
    if display:
        print("\n=== CVXPY box-projection QP ===")
        print(f"is DCP: {problem.is_dcp()}")
        print(f"installed solvers used: {available_solvers}")

    for solver in available_solvers:
        solve_options = {"solver": solver, "verbose": False, "warm_start": True}
        if solver == cp.OSQP:
            solve_options["polishing"] = True
        problem.solve(**solve_options)
        require_solution(problem)
        x_star = np.asarray(x.value).copy()
        dual = np.array([float(c.dual_value) for c in constraints])
        metrics = box_qp_kkt(target.value, x_star, dual)
        np.testing.assert_allclose(x_star, [1.0, 1.0], atol=2e-5)
        np.testing.assert_allclose(problem.value, 5.0, atol=2e-5)
        np.testing.assert_allclose(dual, [4.0, 0.0, 2.0, 0.0], atol=5e-5)
        for name, value in metrics.items():
            assert value < 5e-5, (solver, name, value)
        solutions[solver] = (x_star, dual, float(problem.value))

        if display:
            stats = problem.solver_stats
            print(
                f"{solver}: status={problem.status}, x*={x_star}, "
                f"objective={problem.value:.9g}, dual={dual}, "
                f"iterations={stats.num_iters}, solve_time={stats.solve_time}"
            )

    if cp.CLARABEL in solutions:
        np.testing.assert_allclose(
            solutions[cp.OSQP][0], solutions[cp.CLARABEL][0], atol=2e-5
        )
        np.testing.assert_allclose(
            solutions[cp.OSQP][1], solutions[cp.CLARABEL][1], atol=5e-5
        )

    if display:
        print("\nCVXPY parameter updates with OSQP warm starts:")
    for reference in ([0.25, 0.50], [1.50, 0.50], [3.00, 2.00], [-2.00, -1.50]):
        target.value = np.asarray(reference)
        problem.solve(solver=cp.OSQP, warm_start=True, polishing=True, verbose=False)
        require_solution(problem)
        expected = np.clip(target.value, -1.0, 1.0)
        np.testing.assert_allclose(x.value, expected, atol=2e-5)
        if display:
            print(
                f"target={target.value}, x*={np.asarray(x.value)}, "
                f"iterations={problem.solver_stats.num_iters}"
            )

    target.value = np.array([3.0, 2.0])
    return problem, solutions


def status_examples(*, display=True):
    """Show safe handling of infeasible and unbounded CVXPY problems."""
    if cp.CLARABEL not in cp.installed_solvers():
        raise RuntimeError("CLARABEL is required; reinstall CVXPY with its defaults.")

    x_infeasible = cp.Variable(name="x_infeasible")
    infeasible_problem = cp.Problem(
        cp.Minimize(cp.square(x_infeasible)),
        [x_infeasible <= -1, x_infeasible >= 1],
    )
    infeasible_problem.solve(solver=cp.CLARABEL, verbose=False)
    assert infeasible_problem.status in {cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE}
    assert np.isposinf(infeasible_problem.value)
    assert x_infeasible.value is None

    x_unbounded = cp.Variable(name="x_unbounded")
    unbounded_problem = cp.Problem(
        cp.Minimize(x_unbounded), [x_unbounded <= 0]
    )
    unbounded_problem.solve(solver=cp.CLARABEL, verbose=False)
    assert unbounded_problem.status in {cp.UNBOUNDED, cp.UNBOUNDED_INACCURATE}
    assert np.isneginf(unbounded_problem.value)
    assert x_unbounded.value is None

    if display:
        print("\n=== CVXPY status handling ===")
        print(
            "infeasible problem: "
            f"status={infeasible_problem.status}, value={infeasible_problem.value}, "
            f"x={x_infeasible.value}"
        )
        print(
            "unbounded problem:  "
            f"status={unbounded_problem.status}, value={unbounded_problem.value}, "
            f"x={x_unbounded.value}"
        )
        print("No variable value is read as a solution after either failure status.")

    return infeasible_problem, unbounded_problem


def main():
    np.set_printoptions(precision=6, suppress=True)
    production_lp_example()
    box_qp_example()
    status_examples()


if __name__ == "__main__":
    main()
