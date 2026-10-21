# Tracking MPC: Python Examples

These examples accompany Lecture 8, **Practical Model Predictive Control**,
especially the DC motor tracking example.
`tracking_mpc.py` compares tracking MPC with unconstrained LQR.
`tracking_mpc_disturbance.py` compares nominal and offset-free MPC under an
unknown load, using speed measurements and state observers.
`tracking_mpc_model_error.py` uses the same observer and MPC designs to
compensate for a friction error in the plant model. All three scripts
use CVXPY and OSQP to solve a finite-horizon quadratic program at every
sample and apply only the first optimized input.

## Example

| Lecture topic | Example | Main ideas |
| --- | --- | --- |
| Reference tracking | `tracking_mpc.py` | Compute a steady-state target and regulate state and input deviations |
| Horizon length and terminal cost | `tracking_mpc.py` | Compare short and long horizons with zero or LQR terminal cost |
| Input constraints | `tracking_mpc.py` | Constrain actual motor voltage and compare against unconstrained LQR |
| Disturbance rejection and zero offset | `tracking_mpc_disturbance.py` | Estimate an unknown load, correct the steady-state target, and compare nominal and offset-free MPC |
| Model mismatch and zero offset | `tracking_mpc_model_error.py` | Compensate for increased plant friction using an estimated equivalent disturbance |

The continuous-time motor model is

$$
\dot{x}=\begin{bmatrix}-10&1\\-0.02&-2\end{bmatrix}x
       +\begin{bmatrix}0\\2\end{bmatrix}u,
\qquad y=\begin{bmatrix}1&0\end{bmatrix}x,
$$

where $x=[\omega,\,i]^\mathsf{T}$ contains angular speed and armature current,
and $u$ is the applied voltage. The script discretizes this model using a
zero-order hold with sampling period $T_s=0.05$ s. `tracking_mpc.py` simulates 3 s from
$x_0=[0,\,0]^\mathsf{T}$, tracking the speed reference $r=1$ rad/s.
The corresponding target is $x_s=[1,\,10]^\mathsf{T}$ and $u_s=10.01$ V.

With $\Delta x=x-x_s$ and $\Delta u=u-u_s$, MPC minimizes

$$
J_N=\sum_{i=0}^{N-1}
\left(\Delta x_i^\mathsf{T}Q\Delta x_i
     +\Delta u_i^\mathsf{T}R\Delta u_i\right)
     +\Delta x_N^\mathsf{T}P\Delta x_N,
$$

with $Q=I$, $R=0.1$, and either $P=0$ or the discrete algebraic Riccati
equation solution. By default, $N=4$, the LQR terminal cost is used, and
the actual voltage satisfies $0\leq\Delta u_i+u_s\leq20$ V. The LQR
comparison always uses unconstrained feedback. With the LQR terminal cost
and no input constraints, MPC reproduces the LQR feedback.

## Setup

From this directory, create a virtual environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Run

From this directory, run the default configuration:

```bash
python tracking_mpc.py
```

To reproduce the configurations discussed in the slides:

```bash
# Short horizon, zero terminal cost, no input constraints.
python tracking_mpc.py --horizon 4 --terminal-cost none --unconstrained

# Short horizon, LQR terminal cost, no input constraints.
python tracking_mpc.py --horizon 4 --terminal-cost lqr --unconstrained

# Long horizon, zero terminal cost, no input constraints.
python tracking_mpc.py --horizon 20 --terminal-cost none --unconstrained

# Short horizon, LQR terminal cost, voltage constraints enabled.
python tracking_mpc.py --horizon 4 --terminal-cost lqr

# Long horizon, zero terminal cost, voltage constraints enabled.
python tracking_mpc.py --horizon 20 --terminal-cost none
```

Use `--horizon 10` for the other long-horizon example. Run
`python tracking_mpc.py --help` to see all options.

Each run prints the steady-state target, final speeds, and voltage ranges,
and displays one interactive figure with speed and voltage panels. Time is
shown in seconds, with the reference and enabled voltage bounds marked.
The figure is not saved automatically.

## Load disturbance and zero-offset tracking

Run the load-rejection example from this directory:

```bash
python tracking_mpc_disturbance.py
```

The plant includes an opposing load in the mechanical equation:

$$
\dot{x}=Ax+Bu+Ed,\qquad
E=\begin{bmatrix}-1\\0\end{bmatrix},\qquad d=\tau_L/J.
$$

Here $d$ is torque divided by rotor inertia, in rad/s². It steps from zero
to 2 rad/s² at $t=1$ s. The simulation runs for 5 s with $T_s=0.05$ s,
starting from rest and tracking $r=1$ rad/s. Both MPCs use $N=4$, $Q=I$,
$R=0.1$, the same Riccati terminal weight, and voltage limits $0\leq u\leq20$ V.

Each controller receives only its own measured motor speed and applied
voltage. The simulation is noise-free; true current and load are used only
for plant simulation and evaluation.

| Controller | Observer | Target |
| --- | --- | --- |
| Nominal MPC | Two-state Luenberger observer for speed and current | Fixed at $x_s=[1,10]^\mathsf{T}$, $u_s=10.01$ V |
| Offset-free MPC | Three-state Luenberger observer for speed, current, and constant load | Recomputed at every sample using the estimated load |

The script discretizes the voltage and load channels together, giving
$x_{k+1}=A_dx_k+B_du_k+E_dd_k$. The augmented observer uses

$$
\hat{z}_{k+1}=A_a\hat{z}_k+B_a u_k+L(y_{m,k}-C_a\hat{z}_k),
\qquad
A_a=\begin{bmatrix}A_d&E_d\\0&1\end{bmatrix},\quad
B_a=\begin{bmatrix}B_d\\0\end{bmatrix},\quad
C_a=\begin{bmatrix}C_d&0\end{bmatrix},
$$

where $\hat{z}=[\hat{x}^\mathsf{T},\hat{d}]^\mathsf{T}$ and $C_d$ denotes
the discretized motor output matrix. In the lecture's disturbance notation,
the state-disturbance matrix is $E_d$ and the output-disturbance matrix is zero.
Pole placement sets the nominal observer poles to $0.4,0.5$ and the augmented
observer poles to $0.4,0.5,0.6$. These are predictor-form observers: the
innovation at sample $k$ updates the estimate for sample $k+1$.

The offset-free target satisfies

$$
(I-A_d)x_s-B_du_s=E_d\hat{d},\qquad C_dx_s=r.
$$

Subtracting this equilibrium from the prediction model cancels the estimated
constant load, so the same deviation-coordinate regulation MPC can be used.
Its voltage bounds are updated with the target: $0\leq\Delta u_i+u_s\leq20$.

With the default settings, nominal MPC settles near 0.8954 rad/s, while
offset-free MPC returns to 1 rad/s and estimates the load as 2 rad/s².
Its target approaches $x_s=[1,12]^\mathsf{T}$ and $u_s=12.01$ V.
The nominal observer has model mismatch under load, so its result differs
from a nominal controller supplied with the true full state.

The script prints final tracking errors, voltage ranges, the load estimate,
and the corrected target. One interactive figure shows speed, voltage, and
actual versus estimated load; the load application time is marked in each
panel. The figure is not saved automatically. Change the constants at the
top of `tracking_mpc_disturbance.py` to adjust the experiment.

## Friction model error and zero-offset tracking

Run the model-mismatch example from this directory:

```bash
python tracking_mpc_model_error.py
```

Only the plant's continuous-time friction coefficient changes from
$A_{11}=-10$ to $-11$. The mismatch is present from $t=0$, and there is no
external load. Both controllers retain the nominal coefficient $-10$.
The true plant is discretized separately with a zero-order hold; its
discrete-time state and input matrices therefore both change.

The experiment uses the same settings as the load example: a 5 s simulation
from rest, $T_s=0.05$ s, $r=1$ rad/s, $N=4$, $Q=I$, $R=0.1$, the nominal
Riccati terminal weight, and voltage bounds $0\leq u\leq20$ V. Nominal MPC
uses a two-state observer and a fixed target. Offset-free MPC uses a
three-state observer and updates the target from its disturbance estimate.
Both receive only speed measurements, use the same observer poles as the
load example, and have no knowledge of the changed plant coefficient.

In continuous time, the increased friction can be written as an equivalent
opposing disturbance in the nominal mechanical equation:

$$
\dot{\omega}=-10\omega+i-d_{\mathrm{eq}},\qquad
d_{\mathrm{eq}}=(11-10)\omega.
$$

This term varies during the transient. The observer's constant-disturbance
model represents the mismatch at equilibrium; it does not reproduce the
true transient dynamics exactly. At the desired speed, the equivalent
disturbance is 1 rad/s², and the corrected target is
$x_s=[1,11]^\mathsf{T}$, $u_s=11.01$ V.

With the default settings, nominal MPC settles near 0.9503 rad/s, while
offset-free MPC reaches 1 rad/s and its disturbance estimate approaches
1 rad/s². The script prints the final tracking errors, voltage ranges,
estimated disturbance, equivalent friction term, and corrected target.
One interactive figure shows speed, voltage, and the disturbance estimate
alongside the equivalent friction term for the offset-free plant trajectory.
The figure is not saved automatically. Change `PLANT_A11` at the top of
`tracking_mpc_model_error.py` to adjust the plant friction.
