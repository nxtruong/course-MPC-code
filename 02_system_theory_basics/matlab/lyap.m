%% Discrete-time Lyapunov equation example
clear; clc;

A = [1, 1; 0, 1];
B = [1; 0.5];
K = -[0.1160, 0.5385];
Acl = A + B*K;
Q = eye(2);
assert(all(abs(eig(Acl)) < 1),'The closed-loop system must be stable.');

% dlyap(F,Q) solves F*P*F' - P = -Q, so use F = Acl'.
P = dlyap(Acl',Q);

disp(P)
