"""Discrete-time Lyapunov equation example from the lecture."""

import control as ct
import numpy as np


def main():
    a = np.array([[1.0, 1.0], [0.0, 1.0]])
    b = np.array([[1.0], [0.5]])
    k = -np.array([[0.1160, 0.5385]])
    a_closed_loop = a + b @ k
    q = np.eye(2)

    if np.max(np.abs(np.linalg.eigvals(a_closed_loop))) >= 1.0:
        raise ValueError("The closed-loop system must be stable.")

    # ct.dlyap(F, Q) solves F P F^T - P = -Q.
    p = ct.dlyap(a_closed_loop.T, q)
    np.set_printoptions(precision=4, suppress=True)
    print(p)


if __name__ == "__main__":
    main()
