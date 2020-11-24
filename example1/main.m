% Supplementary material for the paper:
% 'KPC: Learning-Based Model Predictive Control with Deterministic Guarantees'
% Authors: E. T. Maddalena, P. Scharnhorst, Y. Jiang and C. N. Jones
%
% We make use of the MPT3 toolbox, CasADi, and Yalmip 
%
% Example 1

%% System definition
%  The dynamics are given in CSTR_ODE.m 

clear all; clc
close all

global Tsamp xs us

nx = 2;
nu = 1;

% constraints
xmin = [1; 0.5]; 
xmax = [3; 2]; 
H_x = [1 0; -1 0; 0 1; 0 -1];
h_x = [xmax(1); -xmin(1); xmax(2); -xmin(2)];
nh = size(h_x,1);
umin = 3; 
umax = 35;

% equilibrium
xs = [2.1406; 1.0909];
us = 14.19;

% sampling period
Tsamp = 30;

%% Data collection
%  Here we are assuming an uniform noise bound across all samples

% prediction horizon, and data set cardinality
N = 3;  
D = [300 400 600]; 

% noise bound (to be drawn uniformily)
delta_bar = 0.001; 

connected = false; % if true, provide a scalar D
visuals = false;
DATASET = collectData(D, N, xmin, xmax, umin, umax, delta_bar, connected, visuals);

%% Defining the KRR models and bounds
%  One for each state dimension and prediction horizon: total of nx*N
%  CasADi-friendly operations only... 

% extending the variable D if necessary
if size(D,2) == 1, D = D*ones(N,1); end

% some nice lengthscales for our kernel (has to be nx times N)
l = [1.7 3.1 5; 2 27.5 5.6]; 

for i = 1:nx
for n = 1:N
    
    % organizing features and labels
    Z{i,n} = DATASET{i,n}(:,1:end-1);
    Y{i,n} = DATASET{i,n}(:,end);
    
    % defining the kernel function (squared exponential)
    k{i,n} = @(z1,z2) exp(-diag((z1-repmat(z2,size(z1,1),1))*(z1-repmat(z2,size(z1,1),1))') / (2*l(i,n)^2));
     
    % KRR regularizer
    lambda = 0.0000001; 

    % gramian (kernel) matrix 
    jitter = 0.00001;
    K{i,n} = (exp(-dist(Z{i,n},Z{i,n}').^2 / (2*l(i,n)^2))) + jitter*eye(D(n));
    
    % nominal model (eq. 6)
    alpha{i,n} = ((K{i,n}+D(n)*lambda*eye(D(n)))\Y{i,n}); % aux variable
    fhat{i,n} = @(z) exp(-diag((Z{i,n}-repmat(z,size(Z{i,n},1),1))*(Z{i,n}-repmat(z,size(Z{i,n},1),1))') / (2*l(i,n)^2))' * alpha{i,n}; 
    
    % nominal model norm and ground-truth norm estimate
    fHatNor{i,n} = sqrt(Y{i,n}'*(K{i,n}\Y{i,n}));
    Gamma{i,n} = 1.5*fHatNor{i,n};  

    % calculating Delta
    delta = sdpvar(D(n),1);
    constr = -delta_bar <= delta <= delta_bar;
    cost = -(delta'/K{i,n})*delta + 2*(Y{i,n}'/K{i,n})*delta;
    a = optimize(constr, -cost, sdpsettings('solver','mosek','verbose',0));
    delta = value(delta);
    Delta{i,n} = -(delta'/K{i,n})*delta + 2*(Y{i,n}'/K{i,n})*delta;
    
    % bounds
    pow{i,n} = @(z) diag(k{i,n}(z,z) - (k{i,n}(Z{i,n},z)' / K{i,n}) * k{i,n}(Z{i,n},z));                 
    p{i,n}   = @(z) abs((delta_bar*ones(D(n),1)' * abs(K{i,n} \ k{i,n}(Z{i,n},z)))');
    q{i,n}   = @(z) abs((Y{i,n}' / (K{i,n}+(1/(D(n)*lambda))*K{i,n}*K{i,n}) * k{i,n}(Z{i,n},z))');    
    
    bnd{i,n} = @(z) pow{i,n}(z).*sqrt(Gamma{i,n}^2 + Delta{i,n} - fHatNor{i,n}^2) + p{i,n}(z) + q{i,n}(z);
                                
end
end

disp('Done creating models and bounds!')

%% KPC and closed-loop simulation

clear X U LAM J sim
import casadi.*

% enable safety constraints
SAFETY = false;

% total simulation time
sim.Ts = 10;

% some nice initial conditions 
x0 = [1.2; 0.5];
% x0 = [1.3; 1.8]; 
% x0 = [2; 1.8];
% x0 = [1.1; 1.38];
% x0 = [2.8; 0.8];
% x0 = [2.8; 1.8];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% KPC problem definition
% using the LAM variables to eliminate the 1-norm from the
% formulation without adding any conservatism
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

opti = casadi.Opti();

X0  = opti.parameter(2,1); 
X   = opti.variable(nx,N+1);
U   = opti.variable(nu,N);
LAM = opti.variable(nx,nh*N);
J   = 0;

% costs
Q = diag([0.2 1]); 
R = 0.5;
P = [14.46 13.56; 13.56 62.22];

% KPC (no safe set imposed)
opti.subject_to(X(:,1) == X0);   
opti.subject_to(LAM(:) >= 0);
for t = 1:N
    z = [X0' U(:,1:t)];   
    
    opti.subject_to(X(:,t+1) == [fhat{1,t}(z); fhat{2,t}(z)]);   
    opti.subject_to(umin <= U(:,t) <= umax);
    
    %%%%%%%%% Safety constraints %%%%%%%%%
    if SAFETY
        for i = 1:nh
            opti.subject_to(...
                H_x(i,:)*[fhat{1,t}(z); fhat{2,t}(z)] ...
                + ones(nx,1)'*LAM(:,(i+(t-1)*nh)) - h_x(i) <= 0 );
            opti.subject_to(...
                LAM(:,(i+(t-1)*nh)) >= diag([bnd{1,t}(z) bnd{2,t}(z)])*H_x(i,:)' );
            opti.subject_to(...
                LAM(:,(i+(t-1)*nh)) >= -diag([bnd{1,t}(z) bnd{2,t}(z)])*H_x(i,:)' );
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    J = J + ((X(:,t)-xs)'*Q*(X(:,t)-xs) + (U(:,t)-us)'*R*(U(:,t)-us));
end
J = J + (X(:,N+1)-xs)'*P*(X(:,N+1)-xs);

opti.minimize(J);
ops = struct;
ops.ipopt.tol = 1e-3;
opti.solver('ipopt', ops);

%%%%%%%%%%%%%%%%%%%%%
% System simulation
%%%%%%%%%%%%%%%%%%%%%

sim.x = zeros(nx,sim.Ts + 1);
sim.u = zeros(nu,sim.Ts);
sim.x(:,1) = x0;
for t = 1:sim.Ts
    
    % if an init guess is available, use it
    if SAFETY
        try
            opti.set_initial(U, UU{t});
            opti.set_initial(X, XX{t});
            opti.set_initial(LAM, LA{t});
            disp('Warm starting...')
        catch
            disp('Did not warm start...')
        end
    end
   
    % set initial state and solve the problem
    opti.set_value(X0, sim.x(:,t));
    sol = opti.solve();
    
    % extract first optimal move
    sim.u(:,t) = sol.value(U(:,1)); 

    % simulate the continuous-time system (ZOH control)
    [time,x] = ode45(@(time,x) CSTR_ODE(time,x,sim.u(:,t)), [0 Tsamp], sim.x(:,t));
    sim.x(:,t+1) = [x(end,1); x(end,2)];
    
    % collect all OL predictions
    XX{t} = sol.value(X);
    UU{t} = sol.value(U);
    LA{t} = sol.value(LAM);
    DU{t} = sol.value(opti.lam_g);
    for i = 1:N
        z = [XX{t}(:,1)' UU{t}(:,1:i)];
        BB{t}(i+1) = Polyhedron([1 0; -1 0; 0 1; 0 -1],[bnd{1,i}(z); bnd{1,i}(z); bnd{2,i}(z); bnd{2,i}(z)]);
    end

end

disp('Done simulating the system!')

%%%%%%%%%%%%%%%%%%%%%%
% Plotting the results
%%%%%%%%%%%%%%%%%%%%%%

% some neat colors
AZZURRO = [0.8500 0.3250 0.0980];
RIPEORANGE = [0 0.4470 0.7410];

% closed-loop trajectory
figure(1)
plot(sim.x(1,:),sim.x(2,:),'r-x','linewidth',2, 'color', AZZURRO); hold on
plot(x0(1),x0(2),'o','linewidth',2, 'markersize', 8, 'markerfacecolor', AZZURRO ,'color', AZZURRO)
plot(xs(1),xs(2),'o','linewidth',2, 'color', 'k', 'markersize', 8);
axis([xmin(1) xmax(1) xmin(2) xmax(2)])
grid on; set(gcf,'color','w');
X_feas = Polyhedron([1 0; -1 0; 0 1; 0 -1],[xmax(1); -xmin(1); xmax(2); -xmin(2)]);
plot(X_feas, 'color', 'red', 'linewidth', 1.5, 'linestyle', '--', 'alpha', 0.02)

% predictions
figure(2)
plot(x0(1),x0(2),'o','linewidth',2, 'markersize', 8, 'markerfacecolor', RIPEORANGE ,'color', RIPEORANGE); hold on
plot(xs(1),xs(2),'o','linewidth',2, 'color', 'k', 'markersize', 8);
hand = [];
for t = 1:sim.Ts
    plot(XX{t}(1,:),XX{t}(2,:),'-x', 'color', RIPEORANGE, 'linewidth',2);
    for i = 1:N, temp = plot(BB{t}(i+1)+XX{t}(:,i+1),'color',RIPEORANGE,'alpha',0.1); hand = [hand temp]; end
end
for i = 1:(numel(hand)/2), uistack(hand(i),'top'); end
axis([xmin(1) xmax(1) xmin(2) xmax(2)])
grid on; set(gcf,'color','w');
plot(X_feas, 'color', 'blue', 'linewidth', 1.5, 'linestyle', '--', 'alpha', 0.02)

% EOF