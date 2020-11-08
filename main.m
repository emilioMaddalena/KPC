clear 
close all
clc

global T Tsamp xs

T = [0 3000]; 
Tsamp = 30;
nSamp = (T(2) - T(1)) / Tsamp;

nx = 2;
xmin = [0.8; 0.5]; 
xmax = [3; 2]; 
x0 = [1; 1]; 
xs = [2.1406; 1.0909];

nu = 1;
umin = 3;
umax = 35;
us = 14.19;
uex = (umax - umin).*rand(1,nSamp) + umin;

H_x = [1 0; -1 0; 0 1; 0 -1];
h_x = [xmax(1); xmin(1); xmax(2); xmin(2)];
nh = size(h_x,1);

[t,x] = ode45(@(t,x) em_CSTR(t,x,uex), T, x0);

%% Generating the necessary data

noise = [-0.001, 0.001]; 

N = 3;  % prediction horizon
D = [300 400 600]; % the further into the future, the more points we need

connected = false;
visuals = false;
DATASET = collectData(D, N, xmin, xmax, umin, umax, connected, visuals);

%% Defining the KRR models

if size(D,2) == 1, D = D*ones(N,1); end

l = [10 20 40]; 

sigma0 = 0.0001; 
sigmaF0 = 1;
sigmaM0 = [10 20 40 50];

for i = 1:nx
for n = 1:N
    
    % Organizing features and labels
    Z{i,n} = DATASET{i,n}(:,1:end-1);
    Y{i,n} = DATASET{i,n}(:,end);
    
    % Using the GP toolbox to optimize the hyperparams
    gprMdl = fitrgp(Z{i,n},Y{i,n},'KernelFunction','squaredexponential','KernelParameters',[sigmaM0(n);sigmaF0],...
                'Sigma',sigma0,'ConstantSigma',true,'OptimizeHyperparameters',{'KernelScale'},'Standardize',1,'verbose',0);
    gprMdl.KernelInformation.KernelParameters(1)
    
    l(i,n) = gprMdl.KernelInformation.KernelParameters(1); % extract lengthscale
    
    % Defining the kernel function
    k{i,n} = @(z1,z2) exp(-diag((z1-repmat(z2,size(z1,1),1))*(z1-repmat(z2,size(z1,1),1))') / (2*l(i,n)^2));
    
end
end

disp('Learned kernel lengthscales:')
disp(l)

%% Build nominal model and bounds


% Pick an appropriate 'safety factor' in line 94 
% to estimate the ground-truth norm 

for i = 1:nx
for n = 1:N
     
    %%%%%%%%%%%%%%%%
    % nominal model 
    %%%%%%%%%%%%%%%%
    lambda = 0.000001; 

    % Gramian matrix 
    K{i,n} = (exp(-dist(Z{i,n},Z{i,n}').^2 / (2*l(i,n)^2))) + 0.00001*eye(D(n));
    
    % Nominal model
    alpha{i,n} = ((K{i,n}+D(n)*lambda*eye(D(n)))\Y{i,n});
    fhat{i,n} = @(z) exp(-diag((Z{i,n}-repmat(z,size(Z{i,n},1),1))*(Z{i,n}-repmat(z,size(Z{i,n},1),1))') / (2*l(i,n)^2))' * alpha{i,n};
    
    fHatNor{i,n} = sqrt(Y{i,n}'*(K{i,n}\Y{i,n}));
    Gamma{i,n} = 1.2*fHatNor{i,n}; % safety factor here

    %%%%%%%%%%%%%%%%
    % bounds
    %%%%%%%%%%%%%%%%
    
    % Calculating Delta
    delta = sdpvar(D(n),1);
    constr = noise(1) <= delta <= noise(2);
    cost = (delta'/K{i,n})*delta - 2*(Y{i,n}'/K{i,n})*delta;
    a = optimize(constr,cost,sdpsettings('solver','mosek','verbose',0));
    delta = value(delta);
    Delta{i,n} = -(delta'/K{i,n})*delta + 2*(Y{i,n}'/K{i,n})*delta;
    
    sqr_root_term{i,n} = sqrt(Gamma{i,n}^2 + Delta{i,n} - fHatNor{i,n}^2);
    aux_term{i,n} = Y{i,n}' / (K{i,n}+(1/(D(n)*lambda))*K{i,n}*K{i,n});
    bnd{i,n} = @(z) diag(exp(-diag((z-repmat(z,size(z,1),1))*(z-repmat(z,size(z,1),1))') / (2*l(i,n)^2)) ...
                    - (exp(-diag((Z{i,n}-repmat(z,size(Z{i,n},1),1))*(Z{i,n}-repmat(z,size(Z{i,n},1),1))') / (2*l(i,n)^2))'/K{i,n})*exp(-diag((Z{i,n}-repmat(z,size(Z{i,n},1),1))*(Z{i,n}-repmat(z,size(Z{i,n},1),1))') / (2*l(i,n)^2))) ...
                    .* sqr_root_term{i,n} + ...
                    abs((noise(2)*ones(D(n),1)'*abs(K{i,n}\ exp(-diag((Z{i,n}-repmat(z,size(Z{i,n},1),1))*(Z{i,n}-repmat(z,size(Z{i,n},1),1))') / (2*l(i,n)^2)) ) )') ...
                    + abs((aux_term{i,n} * exp(-diag((Z{i,n}-repmat(z,size(Z{i,n},1),1))*(Z{i,n}-repmat(z,size(Z{i,n},1),1))') / (2*l(i,n)^2) ))');

end
end

disp('Done creating models and bounds!')

%% (OPTIONAL) Taking a look at the results

gran = 50;
x1 = linspace(xmin(1),xmax(1),gran);
x2 = linspace(xmin(2),xmax(2),gran);

[X1,X2] = meshgrid(x1,x2);

% 1 step
for j = 1:size(X1(:))
    Fhat(j) = fhat{2,1}([X1(j) X2(j) 25]);
end
Fhat = reshape(Fhat,size(X1));

[x1next,x2next] = groundTruth(X1(:),X2(:),25);
F = reshape(x2next,size(X1));

subplot(1,2,1)
surf(X1,X2,F)

subplot(1,2,2)
surf(X1,X2,Fhat)

%% (OPTIONAL) Taking a look at the size of the bounds for an open-loop seq

clear x F

uex = (umax - umin).*rand(1,N) + umin;
x(1,:) = x0;

x(1,:) = Z{1,1}(1,1:2)';
uex = Z{1,1}(1:N,3)';

F = zeros(N,nx);

figure
plot(x0(1),x0(2),'ro','markersize',10,'linewidth',2); hold on
for t = 1:N
    
    z = [x0' uex(1:t)];
    
    x(t+1,1) = fhat{1,t}(z);
    x(t+1,2) = fhat{2,t}(z);
    
    F(t,:) = [fhat{1,t}(z) fhat{2,t}(z)];
    B(t) = Polyhedron([1 0; -1 0; 0 1; 0 -1],[bnd{1,t}(z); bnd{1,t}(z); bnd{2,t}(z); bnd{2,t}(z)]);
    
    plot(B(t)+F(t,:)','color','blue','alpha',0.1); hold on; grid on;
    plot(F(t,1),F(t,2),'ko','markersize',10,'linewidth',2); grid on
    
end

%% KPC

clear X U LAM J opti sim

import casadi.*
opti = casadi.Opti();

X0  = opti.parameter(2,1); 
X   = opti.variable(nx,N+1);
U   = opti.variable(nu,N);
LAM = opti.variable(nx,nh*N);
J   = 0;

% Defining the MPC controller
Q = diag([0.2 1]); 
P = [14.46 13.56; 13.56 62.22];
R = 0.5;
opti.subject_to(X(:,1) == X0);   
opti.subject_to(LAM(:) >= 0);
for t = 1:N
    z = [X0' U(:,1:t)];   
    
    opti.subject_to(X(:,t+1) == [fhat{1,t}(z); fhat{2,t}(z)]);   
    opti.subject_to(umin <= U(:,t) <= umax);
    
    %%%% Safety constraints %%%%%
%     for i = 1:nh
%         opti.subject_to(...
%             H_x(i,:)*[fhat{1,t}(z); fhat{2,t}(z)] ...
%             + ones(nx,1)'*LAM(:,(i+(t-1)*nh)) - h_x(i) <= 0 );
%         opti.subject_to(...
%             LAM(:,(i+(t-1)*nh)) >= diag([bnd{1,t}(z) bnd{2,t}(z)])*H_x(i,:)' );
%         opti.subject_to(...
%             LAM(:,(i+(t-1)*nh)) >= -diag([bnd{1,t}(z) bnd{2,t}(z)])*H_x(i,:)' );
%     end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    J = J + ((X(:,t)-xs)'*Q*(X(:,t)-xs) + (U(:,t)-us)'*R*(U(:,t)-us));
end
opti.subject_to(xmin <= X(:,N) <= xmax);
J = J + (X(:,N+1)-xs)'*P*(X(:,N+1)-xs);
opti.minimize(J);
ops = struct;
ops.ipopt.tol = 1e-3;
opti.solver('ipopt', ops);

%%%%%%%%%%%%%%%%%%%%%
% Simulation
%%%%%%%%%%%%%%%%%%%%%
sim.Ts = 10;

x0 = [2.3; 1.4];
%x0 = [1.8; 1.4]; 
%x0 = [1.2; 1]; 
%x0 = [2.6; 0.7]; 
%x0 = [1.4; 1.4]; 
%x0 = [2.8; 1.1];
x0 = [1.8; 0.5];

sim.x = zeros(nx,sim.Ts + 1);
sim.u = zeros(nu,sim.Ts);
sim.x(:,1) = x0;
for t = 1:sim.Ts
    
    opti.set_value(X0, sim.x(:,t));
    sol = opti.solve();
    sim.u(:,t) = sol.value(U(:,1)); 
    
    [time,x] = ode45(@(time,x) em_CSTR(time,x,sim.u(:,t)), [0 Tsamp], sim.x(:,t));
    sim.x(:,t+1) = [x(end,1); x(end,2)];
    
    % Collect OL predictions
    XX{t} = sol.value(X);
    UU{t} = sol.value(U);
    for i = 1:N
        z = [XX{t}(:,1)' UU{t}(:,1:i)];
        BB{t}(i+1) = Polyhedron([1 0; -1 0; 0 1; 0 -1],[bnd{1,i}(z); bnd{1,i}(z); bnd{2,i}(z); bnd{2,i}(z)]);
    end
    % Warm start the next iteration
    opti.set_initial(U, sol.value(U));
    opti.set_initial(X, sol.value(X));

end

disp('Done simulating the system!')

%% Plotting

% Plotting closed-loop trajectory
plot(sim.x(1,:),sim.x(2,:),'r-*','linewidth',2); hold on
plot(x0(1),x0(2),'ro','linewidth',2)
plot(xs(1),xs(2),'rx','linewidth',2)
 
% Plotting predicted trajectories and boxes
for t = 1:1
    pause
    plot(XX{t}(1,:),XX{t}(2,:),'b-*','linewidth',2);
    for i = 1:N, plot(BB{t}(i+1)+XX{t}(:,i+1),'color','blue','alpha',0.1); end
    axis([xmin(1) xmax(1) xmin(2) xmax(2)])
    grid on
end
