"""Direct OSQP examples for Lecture 4: Convex Optimization.

The examples intentionally expose the matrix bookkeeping required by a direct
solver interface.  All certificate checks are numerical checks subject to the
solver tolerances; they are not exact symbolic proofs.
"""

from __future__ import annotations

import numpy as np
import osqp
from scipy import sparse


SOLVED_STATUSES = {"solved", "solved inaccurate"}


def box_qp_data(target: np.ndarray):
    """Return OSQP data for projecting ``target`` onto the unit box.

    The original objective is ||x - target||_2^2.  OSQP receives the same
    objective without the constant target.T @ target:

        (1/2) x.T P x + q.T x,  P = 2 I,  q = -2 target.

    Constraint rows follow the lecture order: x1, -x1, x2, -x2 <= 1.
    """
    target = np.asarray(target, dtype=float).reshape(2)
    p = 2.0 * sparse.eye(2, format="csc")
    q = -2.0 * target
    a = sparse.csc_matrix(
        np.array(
            [
                [1.0, 0.0],
                [-1.0, 0.0],
                [0.0, 1.0],
                [0.0, -1.0],
            ]
        )
    )
    lower = np.full(4, -np.inf)
    upper = np.ones(4)
    constant = float(target @ target)
    return p, q, a, lower, upper, constant


def make_solver(p, q, a, lower, upper, *, polishing=True):
    """Set up an accurate, quiet OSQP solver instance."""
    solver = osqp.OSQP()
    solver.setup(
        P=p,
        q=q,
        A=a,
        l=lower,
        u=upper,
        verbose=False,
        polishing=polishing,
        eps_abs=1e-9,
        eps_rel=1e-9,
    )
    return solver


def require_solution(result):
    """Reject failed solves before any primal or dual values are consumed."""
    status = result.info.status.lower()
    if status not in SOLVED_STATUSES:
        raise RuntimeError(f"OSQP did not solve the problem: {result.info.status}")


def box_kkt_metrics(p, q, a, upper, constant, x, dual):
    """Return KKT and primal-dual-gap residuals for upper-bound inequalities."""
    gradient = np.asarray(p @ x + q).reshape(-1)
    constraint_values = np.asarray(a @ x - upper).reshape(-1)
    stationarity = np.asarray(gradient + a.T @ dual).reshape(-1)

    # For g(x) = A x - u <= 0, the lecture convention is dual >= 0.
    primal_violation = np.maximum(constraint_values, 0.0)
    dual_violation = np.maximum(-dual, 0.0)
    complementarity = dual * constraint_values

    # The dual function for this strictly convex QP can be evaluated directly.
    shifted_gradient = np.asarray(q + a.T @ dual).reshape(-1)
    p_inverse_term = np.linalg.solve(p.toarray(), shifted_gradient)
    dual_objective = (
        -0.5 * float(shifted_gradient @ p_inverse_term)
        - float(upper @ dual)
        + constant
    )
    primal_objective = (
        0.5 * float(x @ (p @ x)) + float(q @ x) + constant
    )

    return {
        "primal_violation": float(np.linalg.norm(primal_violation, np.inf)),
        "dual_violation": float(np.linalg.norm(dual_violation, np.inf)),
        "stationarity": float(np.linalg.norm(stationarity, np.inf)),
        "complementarity": float(np.linalg.norm(complementarity, np.inf)),
        "primal_objective": primal_objective,
        "dual_objective": dual_objective,
        "duality_gap": primal_objective - dual_objective,
    }


def solve_box_projection(target=(3.0, 2.0), *, display=True):
    """Solve and verify the lecture's box-projection QP."""
    target = np.asarray(target, dtype=float)
    p, q, a, lower, upper, constant = box_qp_data(target)
    solver = make_solver(p, q, a, lower, upper)
    result = solver.solve()
    require_solution(result)

    x = np.asarray(result.x)
    dual = np.asarray(result.y)
    metrics = box_kkt_metrics(p, q, a, upper, constant, x, dual)

    expected_x = np.clip(target, -1.0, 1.0)
    np.testing.assert_allclose(x, expected_x, atol=2e-6)
    for name in (
        "primal_violation",
        "dual_violation",
        "stationarity",
        "complementarity",
        "duality_gap",
    ):
        assert abs(metrics[name]) < 2e-6, (name, metrics[name])

    if np.allclose(target, [3.0, 2.0]):
        np.testing.assert_allclose(x, [1.0, 1.0], atol=2e-6)
        np.testing.assert_allclose(dual, [4.0, 0.0, 2.0, 0.0], atol=2e-5)
        np.testing.assert_allclose(metrics["primal_objective"], 5.0, atol=2e-6)

    if display:
        print("=== Direct OSQP: projection onto the unit box ===")
        print(f"status:              {result.info.status}")
        print(f"x*:                  {x}")
        print(f"dual y*:             {dual}")
        print(f"original objective:  {metrics['primal_objective']:.9g}")
        print(f"OSQP objective:      {result.info.obj_val:.9g} (constant removed)")
        print(f"primal residual:     {result.info.prim_res:.3e}")
        print(f"dual residual:       {result.info.dual_res:.3e}")
        print(f"duality gap:         {result.info.duality_gap:.3e}")
        print(f"iterations:          {result.info.iter}")
        print(f"solve time:          {result.info.solve_time:.3e} s")
        print("KKT residuals:")
        for key in (
            "primal_violation",
            "dual_violation",
            "stationarity",
            "complementarity",
            "duality_gap",
        ):
            print(f"  {key:20s} {metrics[key]:.3e}")

    return solver, result, metrics


def repeated_box_projections(*, display=True):
    """Update q and reuse one solver, as is common in online MPC."""
    targets = np.array(
        [
            [0.25, 0.50],
            [1.50, 0.50],
            [3.00, 2.00],
            [-2.00, -1.50],
        ]
    )
    p, q, a, lower, upper, _ = box_qp_data(targets[0])
    # Polishing is unnecessary for the interior reference and can print an
    # informational message even with verbose=False, so leave it off here.
    solver = make_solver(p, q, a, lower, upper, polishing=False)
    records = []

    if display:
        print("\n=== Reusing one OSQP instance for changing references ===")
    for target in targets:
        q = -2.0 * target
        solver.update(q=q)
        result = solver.solve()
        require_solution(result)
        x = np.asarray(result.x)
        dual = np.asarray(result.y)
        expected = np.clip(target, -1.0, 1.0)
        np.testing.assert_allclose(x, expected, atol=2e-6)

        constant = float(target @ target)
        metrics = box_kkt_metrics(p, q, a, upper, constant, x, dual)
        assert metrics["stationarity"] < 2e-6
        assert metrics["complementarity"] < 2e-6
        records.append((target.copy(), x.copy(), dual.copy(), result.info.iter))

        if display:
            print(
                f"target={target}, x*={x}, dual={dual}, "
                f"iterations={result.info.iter}"
            )

    return records


def primal_infeasibility_example(*, display=True):
    """Detect x <= -1 and x >= 1 and verify OSQP's certificate."""
    p = sparse.csc_matrix([[2.0]])
    q = np.array([0.0])
    a = sparse.csc_matrix([[1.0], [1.0]])
    lower = np.array([-np.inf, 1.0])
    upper = np.array([-1.0, np.inf])
    result = make_solver(p, q, a, lower, upper).solve()

    assert "primal infeasible" in result.info.status.lower()
    certificate = np.asarray(result.prim_inf_cert)
    at_v = np.asarray(a.T @ certificate).reshape(-1)

    positive = (certificate > 0.0) & np.isfinite(upper)
    negative = (certificate < 0.0) & np.isfinite(lower)
    separating_value = float(
        upper[positive] @ certificate[positive]
        + lower[negative] @ certificate[negative]
    )
    bound_sign_violation = float(
        max(
            np.max(np.maximum(certificate[np.isposinf(upper)], 0.0), initial=0.0),
            np.max(np.maximum(-certificate[np.isneginf(lower)], 0.0), initial=0.0),
        )
    )
    at_v_norm = float(np.linalg.norm(at_v, np.inf))
    assert at_v_norm < 2e-5
    assert bound_sign_violation < 2e-5
    assert separating_value < -1e-5

    if display:
        print("\n=== OSQP primal-infeasibility certificate ===")
        print(f"status:                         {result.info.status}")
        print(f"certificate v:                  {certificate}")
        print(f"||A^T v||_inf:                  {at_v_norm:.3e}")
        print(f"infinite-bound sign violation:  {bound_sign_violation:.3e}")
        print(f"u^T v_+ + l^T v_-:             {separating_value:.3e} < 0")

    return result, at_v_norm, bound_sign_violation, separating_value


def dual_infeasibility_example(*, display=True):
    """Detect the unbounded problem min x subject to x <= 0."""
    p = sparse.csc_matrix((1, 1))
    q = np.array([1.0])
    a = sparse.csc_matrix([[1.0]])
    lower = np.array([-np.inf])
    upper = np.array([0.0])
    result = make_solver(p, q, a, lower, upper).solve()

    assert "dual infeasible" in result.info.status.lower()
    direction = np.asarray(result.dual_inf_cert)
    p_s_norm = float(np.linalg.norm(np.asarray(p @ direction), np.inf))
    q_s = float(q @ direction)
    a_s = np.asarray(a @ direction).reshape(-1)
    direction_violation = float(np.linalg.norm(np.maximum(a_s, 0.0), np.inf))
    assert p_s_norm < 2e-5
    assert q_s < -1e-5
    assert direction_violation < 2e-5

    if display:
        print("\n=== OSQP dual-infeasibility (unboundedness) certificate ===")
        print(f"status:                         {result.info.status}")
        print(f"certificate s:                  {direction}")
        print(f"||P s||_inf:                    {p_s_norm:.3e}")
        print(f"q^T s:                          {q_s:.3e} < 0")
        print(f"upper-bound direction violation:{direction_violation:.3e}")

    return result, p_s_norm, q_s, direction_violation


def main():
    np.set_printoptions(precision=6, suppress=True)
    solve_box_projection()
    repeated_box_projections()
    primal_infeasibility_example()
    dual_infeasibility_example()


if __name__ == "__main__":
    main()
