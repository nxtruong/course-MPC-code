"""Shared model, simulation, and plotting utilities for the MPC examples."""

from __future__ import annotations

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np


A = np.array([[2.0, -1.0], [1.0, 0.2]])
B = np.array([[1.0], [0.0]])
Q = np.eye(2)
R = 2.0
HORIZON = 7
INITIAL_STATE = np.array([3.0, 1.0])
SIMULATION_STEPS = 15


def simulate(controller: Callable[[np.ndarray], np.ndarray]):
    """Apply the first optimized input at each sample, as in the MATLAB code."""
    states = np.empty((SIMULATION_STEPS + 1, A.shape[0]))
    inputs = np.empty(SIMULATION_STEPS)
    states[0] = INITIAL_STATE

    for k in range(SIMULATION_STEPS):
        optimal_inputs = controller(states[k])
        inputs[k] = optimal_inputs[0]
        states[k + 1] = A @ states[k] + B[:, 0] * inputs[k]

    return states, inputs


def plot_trajectories(states: np.ndarray, inputs: np.ndarray, form: str):
    """Reproduce the two-panel state and control plot in the MATLAB examples."""
    fig, (state_axis, input_axis) = plt.subplots(2, 1, figsize=(8, 6))
    state_axis.step(
        np.arange(states.shape[0]), states[:, 0], where="post", label="$x_1$"
    )
    state_axis.step(
        np.arange(states.shape[0]), states[:, 1], where="post", label="$x_2$"
    )
    state_axis.set(title=f"States ({form} prediction form)", ylabel="State")
    state_axis.legend()

    input_axis.step(np.arange(inputs.size), inputs, where="post", label="$u$")
    input_axis.set(title="Control input", xlabel="Time step", ylabel="Input")
    input_axis.legend()

    for axis in (state_axis, input_axis):
        axis.grid(True, alpha=0.35)
    fig.tight_layout()

