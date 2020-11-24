function DATASET = collectData(D, N, xmin, xmax, umin, umax, delta_bar, connected, plts)
    % [DATASET_x1, DATASET_x2]
    global Tsamp xs
    
    if ~exist('connected')
        connected = true;
    end
    
    if ~exist('plts')
        plts = false;
    end

    if connected
    % One datum at a time, simulate for the longest horizon and 
    % place each segment into its dataset
    
        if size(D,2) ~= 1, error('Connected data requires D to be scalar...'); end
        
        x0 = (xmax - xmin).*rand(2,D) + xmin;
        u  = (umax - umin).*rand(1,N,D) + umin;
        
        T = [0 N*Tsamp];
        time = linspace(T(1),T(2),N+1);
        x_steps = zeros(N+1,2);

        for d = 1:D

            % Integrating the system
            [t,x] = ode45(@(t,x) CSTR_ODE(t,x,u(1,:,d)), T, x0(:,d));

            % Extracting x0
            idx = find(t>=time(1), 1, 'first');
            x_steps(1,:) = x(idx,:);

            for n = 2:N+1

                idx = find(t>=time(n), 1, 'first');
                x_steps(n,:) = x(idx,:);

                if plts 
                    figure(n-1)
                    plot(x_steps(1,1),x_steps(1,2),'ko','markersize',8,'linewidth',2); hold on; grid on
                    plot(x_steps(n,1),x_steps(n,2),'kx','markersize',8,'linewidth',2);
                    plot(x_steps(1:n,1), x_steps(1:n,2),'-b','linewidth',2);
                    plot(xs(1),xs(2),'ro','markersize',15,'linewidth',2)
                    axis(1*[xmin(1) xmax(1) xmin(2) xmax(2)])
                    title(['Data collection for ' num2str(n-1) ' control moves!'])
                end
                
                % adding noise
                del = rand(2,1) * 2*delta_bar - delta_bar;

                DATASET{1,n-1}(d,:) = [x_steps(1,:) u(1,1:n-1,d) (x_steps(n,1) + del(1))];
                DATASET{2,n-1}(d,:) = [x_steps(1,:) u(1,1:n-1,d) (x_steps(n,2) + del(2))];
            end
        end
        
     elseif ~connected
     % Fixes the horizon, build entire dataset for it, increases horizon
     
        if size(D,2) == 1, D = D*ones(N,1); end
                
        for n = 1:N
    
            T = [0 n*Tsamp];
            time = linspace(T(1),T(2),n+1);
            x_steps = zeros(n,2);

            for d = 1:D(n)
                
                x0 = (xmax - xmin).*rand(2,1) + xmin;
                u  = (umax - umin).*rand(1,n) + umin;

                [t,x] = ode45(@(t,x) CSTR_ODE(t,x,u), T, x0);

                for i = 1:n+1
                    idx = find(t>=time(i), 1, 'first');
                    x_steps(i,:) = x(idx,:);
                end

                if plts
                    figure(n)
                    plot(x_steps(1,1),x_steps(1,2),'ko','markersize',5,'linewidth',1.5); hold on; grid on 
                    plot(x_steps(end,1),x_steps(end,2),'kx','markersize',5,'linewidth',1.5);
                    plot(x_steps(:,1), x_steps(:,2),'-b','linewidth',1)
                    axis(1*[xmin(1) xmax(1) xmin(2) xmax(2)])
                    title(['Data collection for ' num2str(n) ' control moves!'])
                end
                
                % adding noise
                del = rand(2,1) * 2*delta_bar - delta_bar;
                
                DATASET{1,n}(d,:) = [x_steps(1,:) u (x_steps(end,1) + del(1))];
                DATASET{2,n}(d,:) = [x_steps(1,:) u (x_steps(end,2) + del(2))];
            end

        end
    end
     
    disp('Done collecting data!')
end