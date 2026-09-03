# Pendulum Kalman Filter

This example simulates a nonlinear continuous-time pendulum and estimates its
angle and angular velocity with a discrete-time linear Kalman filter. The
filter model is obtained by linearizing at

$$
x_s = [\pi/4,\ 0]^T, \qquad
u_s = (g/l)\sin(\pi/4),
$$

then applying exact zero-order-hold discretization. Only the pendulum angle is
measured. The Kalman filter is implemented directly using the
predictor--corrector equations from the lecture.

The piecewise signal in the plot is the deviation input $\Delta u$. The
nonlinear plant receives the total input $u_s + \Delta u$.

## Setup

From this directory, create a virtual environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Run

Open `pendulum_kf.ipynb` in Jupyter, VS Code, or Google Colab and run the
cells in order. Its first code cell optionally installs the required libraries
for Colab.

The equivalent script can be run from a terminal:

```bash
python pendulum_kf.py
```

The script prints the angle and angular-velocity estimation RMSE and displays
the actual states, Kalman estimates, noisy angle measurements, and input. Set
`ts` near the start of `main()` to experiment with the sampling time.
