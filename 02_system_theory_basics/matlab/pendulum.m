%% Nonlinear continuous-time pendulum
clear; close all; clc;

% The input u = T_c/(m*l^2) is the normalized external torque.
g = 10;                         % gravitational acceleration [m/s^2]
l = 1;                          % pendulum length [m]
Ts = 0.01;                      % sampling time [s]

f = @(x,u) [x(2); -(g/l)*sin(x(1)) + u];
uDecay = @(t) 0.1*exp(-0.3*t).*sin(2*t);

tSim = 0:Ts:10;
x0 = [deg2rad(15); 0];
[tNonlin,xNonlin] = ode45(@(t,x) f(x,uDecay(t)), tSim, x0);
yNonlin = xNonlin(:,1);         % measured output: y = theta

figure('Name','Nonlinear continuous-time pendulum');
tiledlayout(2,1);
nexttile;
plot(tNonlin,uDecay(tNonlin),'LineWidth',1.2);
grid on; ylabel('u [rad/s^2]');
title('Decaying sinusoidal input');
nexttile;
plot(tNonlin,rad2deg(yNonlin),'LineWidth',1.2);
grid on; xlabel('Time [s]'); ylabel('\theta [deg]');
title('Nonlinear pendulum output');

%% Linearization
% At an equilibrium (xs,us), the general deviation model is
%   Delta xdot = Ac(xs)*Delta x + Bc*Delta u,
%   Delta y    = C*Delta x + D*Delta u.
Ac = @(xs) [0, 1; -(g/l)*cos(xs(1)), 0];
Bc = [0; 1];
C = [1, 0];
D = 0;

% Case 1: equilibrium at theta = pi/4.
xs1 = [pi/4; 0];
us1 = (g/l)*sin(pi/4);
Ac1 = Ac(xs1);

% Case 2: equilibrium at the origin with feedback u = -x2.
xs2 = [0; 0];
uCase2 = @(x) -x(2);
us2 = uCase2(xs2);
Kfb = [0, -1];
Acl2 = Ac(xs2) + Bc*Kfb;

tClosedLoop = 0:Ts:20;
x0Small = [deg2rad(5); deg2rad(2)];
[tNlCl,xNlCl] = ode45(@(t,x) f(x,uCase2(x)), tClosedLoop, x0Small);
[tLinCl,xLinCl] = ode45(@(t,x) Acl2*x, tClosedLoop, x0Small-xs2);
xLinCl = xLinCl + xs2.';

figure('Name','Closed-loop nonlinear and linearized models');
tiledlayout(2,1);
nexttile;
plot(tNlCl,rad2deg(xNlCl(:,1)),'LineWidth',1.2); hold on;
plot(tLinCl,rad2deg(xLinCl(:,1)),'--','LineWidth',1.2);
grid on; ylabel('x_1 [deg]');
legend('Nonlinear','Linearized','Location','best');
title('Case 2: u = -x_2');
nexttile;
plot(tNlCl,rad2deg(xNlCl(:,2)),'LineWidth',1.2); hold on;
plot(tLinCl,rad2deg(xLinCl(:,2)),'--','LineWidth',1.2);
grid on; xlabel('Time [s]'); ylabel('x_2 [deg/s]');
legend('Nonlinear','Linearized','Location','best');

%% Euler and exact zero-order-hold discretization
% Nonlinear Euler model:
%   x(k+1) = x(k) + Ts*[x2(k); -(g/l)*sin(x1(k)) + u(k)].

% sampling time for discretization [s]
% Try small (0.01) and large (0.1) to see the effect
Ts = 0.1;

sysC1 = ss(Ac1,Bc,C,D);
sysD1 = c2d(sysC1,Ts,'zoh');
Ad1 = sysD1.A;
Bd1 = sysD1.B;

tCompare = 0:Ts:4;
du = arrayfun(@pulseInput,tCompare);  % input deviation from us1

xCt = zeros(2,numel(tCompare));
xEuler = zeros(2,numel(tCompare));
dxLinear = zeros(2,numel(tCompare));
xCt(:,1) = xs1;
xEuler(:,1) = xs1;

odeOptions = odeset('RelTol',1e-9,'AbsTol',1e-11);
for k = 1:numel(tCompare)-1
    uTotal = us1 + du(k);

    % Integrate each zero-order-hold interval separately.
    tSpan = tCompare(k:k+1);
    [~,xInterval] = ode45(@(t,x) f(x,uTotal),tSpan,xCt(:,k),odeOptions);
    xCt(:,k+1) = xInterval(end,:).';

    xEuler(:,k+1) = xEuler(:,k) + Ts*f(xEuler(:,k),uTotal);
    dxLinear(:,k+1) = Ad1*dxLinear(:,k) + Bd1*du(k);
end
xLinear = xs1 + dxLinear;

figure('Name','Continuous-time and discrete-time comparison');
tiledlayout(3,1);
nexttile;
plot(tCompare,rad2deg(xCt(1,:)),'LineWidth',1.2); hold on;
plot(tCompare,rad2deg(xEuler(1,:)),'--','LineWidth',1.2);
plot(tCompare,rad2deg(xLinear(1,:)),':','LineWidth',1.5);
grid on; ylabel('x_1 [deg]');
legend('Nonlinear CT','Nonlinear Euler','Linear exact ZOH', ...
    'Location','best');
title('Case 1 models about x_s = [\pi/4, 0]^T');
nexttile;
plot(tCompare,rad2deg(xCt(2,:)),'LineWidth',1.2); hold on;
plot(tCompare,rad2deg(xEuler(2,:)),'--','LineWidth',1.2);
plot(tCompare,rad2deg(xLinear(2,:)),':','LineWidth',1.5);
grid on; ylabel('x_2 [deg/s]');
legend('Nonlinear CT','Nonlinear Euler','Linear exact ZOH', ...
    'Location','best');
nexttile;
stairs(tCompare,du,'LineWidth',1.2);
grid on; xlabel('Time [s]'); ylabel('\Delta u [rad/s^2]');

function u = pulseInput(t)
if t < 1
    u = 0.1;
elseif t < 2
    u = -0.1;
else
    u = 0;
end
end
