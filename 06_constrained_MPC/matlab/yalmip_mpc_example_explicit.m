% An example of MPC implementation and simulation using YALMIP
% Explicit form

%% Clear environment and system definition
yalmip('clear')
clear all

% Model data
A = [2 -1;1 0.2];
B = [1;0];
nx = 2; % Number of states
nu = 1; % Number of inputs

% MPC data
Q = eye(2);
R = 2;
N = 7;

% Initial state
xinit = [3;1];
duration = 15;  % simulation in time steps

%% Implement MPC using the Explicit Prediction Form
u = sdpvar(repmat(nu,1,N),repmat(1,1,N));
x0 = sdpvar(nx,1);

constraints = [];
objective = 0;
x = x0;
for k = 1:N
    objective = objective + norm(Q*x,1) + norm(R*u{k},1);
    constraints = [constraints, -1 <= u{k} <= 1];
    x = A*x + B*u{k};
    % Constrain the successor state x_k; x_0 is the given parameter.
    constraints = [constraints, -5 <= x <= 5];
end

ops = sdpsettings('verbose', 2);
controller = optimizer(constraints,objective,ops,x0,u{1});

%% Run simulation
x = xinit;    % Set the initial state

% State and control input histories
u_trajectory = [];
x_trajectory = xinit';

% Main simulation loop
for i = 1:duration
    U = controller{x};  % MPC returns optimal control
    x = A*x + B*U(1);  % Apply the first control

    % Save the applied control over time
    u_trajectory = [u_trajectory; U(1)'];
    % Save the state over time
    x_trajectory = [x_trajectory; x'];
end

%% Plot simulation results
subplot(2,1,1);
stairs(0:duration, x_trajectory, 'LineWidth', 1);
title('States');
legend('x_1', 'x_2');

subplot(2,1,2);
stairs(0:(duration-1), u_trajectory, 'LineWidth', 1);
title('Control input');
legend('u');
