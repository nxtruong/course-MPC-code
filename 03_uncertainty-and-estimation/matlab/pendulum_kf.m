%% Kalman filter for a pendulum linearized about a nonzero equilibrium
clear; close all; clc;
rng(4);                         % reproducible noise realization

%% Nonlinear continuous-time pendulum
% The input u = T_c/(m*l^2) is the normalized external torque.
g = 10;                         % gravitational acceleration [m/s^2]
l = 1;                          % pendulum length [m]
Ts = 0.01;                      % adjustable sampling time [s]

f = @(x,u) [x(2); -(g/l)*sin(x(1)) + u];

%% Linearization and exact zero-order-hold discretization
xs = [pi/4; 0];
us = (g/l)*sin(xs(1));

Ac = [0, 1; -(g/l)*cos(xs(1)), 0];
Bc = [0; 1];
C = [1, 0];                     % only the angle is measured
D = 0;

sysD = c2d(ss(Ac,Bc,C,D),Ts,'zoh');
A = sysD.A;
B = sysD.B;

% The discrete deviation model used by the filter is
%   dx(k+1) = A*dx(k) + B*du(k) + eps1(k),
%   dy(k)   = C*dx(k) + eps2(k).

%% Known Gaussian disturbance models
R1 = diag([deg2rad(0.02)^2, deg2rad(0.05)^2]);
R2 = deg2rad(1.0)^2;
L1 = chol(R1,'lower');

xHat0 = xs + deg2rad([5; -3]);  % deliberately imperfect initial estimate
P0 = diag(deg2rad([5; 5]).^2);

%% Simulate the nonlinear plant and run the Kalman filter
tFinal = 5;
t = 0:Ts:tFinal;
Nt = numel(t);
du = arrayfun(@pulseInput,t);   % requested input deviation about us

x = zeros(2,Nt);
y = zeros(1,Nt);
xHat = zeros(2,Nt);
x(:,1) = xs;

% Use the first measurement to correct the initial prior estimate.
y(1) = C*x(:,1) + sqrt(R2)*randn;
S = C*P0*C.' + R2;
K = (P0*C.')/S;
xHat(:,1) = xHat0 + K*(y(1) - C*xHat0);
I_KC = eye(2) - K*C;
P = I_KC*P0*I_KC.' + K*R2*K.';

odeOptions = odeset('RelTol',1e-9,'AbsTol',1e-11);
for k = 1:Nt-1
    % The nonlinear plant receives the total input us + du(k).
    uTotal = us + du(k);
    [~,xInterval] = ode45(@(~,xk) f(xk,uTotal), ...
        t(k:k+1),x(:,k),odeOptions);
    x(:,k+1) = xInterval(end,:).' + L1*randn(2,1);
    y(k+1) = C*x(:,k+1) + sqrt(R2)*randn;

    % Predict in deviation variables, then return to absolute state units.
    dxPred = A*(xHat(:,k) - xs) + B*du(k);
    xPred = xs + dxPred;
    PPred = A*P*A.' + R1;

    % Measurement correction and Joseph covariance update.
    S = C*PPred*C.' + R2;
    K = (PPred*C.')/S;
    xHat(:,k+1) = xPred + K*(y(k+1) - C*xPred);
    I_KC = eye(2) - K*C;
    P = I_KC*PPred*I_KC.' + K*R2*K.';
    P = (P + P.')/2;
end

%% Compare the sampled true and estimated states
figure('Name','Pendulum Kalman filter');
tiledlayout(3,1);

nexttile;
plot(t,rad2deg(x(1,:)),'k','LineWidth',1.2); hold on;
plot(t,rad2deg(xHat(1,:)),'--','LineWidth',1.2);
plot(t,rad2deg(y),'.','Color',[0.65 0.75 0.90],'MarkerSize',5);
grid on; ylabel('\theta [deg]');
legend('Actual','Kalman estimate','Measurement','Location','best');
title('Nonlinear pendulum with an LTI Kalman filter');

nexttile;
plot(t,rad2deg(x(2,:)),'k','LineWidth',1.2); hold on;
plot(t,rad2deg(xHat(2,:)),'--','LineWidth',1.2);
grid on; ylabel('\omega [deg/s]');
legend('Actual','Kalman estimate','Location','best');

nexttile;
stairs(t,du,'LineWidth',1.2);
grid on; xlabel('Time [s]'); ylabel('\Delta u [rad/s^2]');
ylim([-0.12 0.12]);

rmse = sqrt(mean((xHat - x).^2,2));
fprintf('State-estimation RMSE: %.3f deg, %.3f deg/s\n', ...
    rad2deg(rmse(1)),rad2deg(rmse(2)));

function u = pulseInput(t)
if t < 1
    u = 0.1;
elseif t < 2
    u = -0.1;
else
    u = 0;
end
end
