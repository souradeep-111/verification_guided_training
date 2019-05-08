 Ts = 0.2;  % Sample Time
N = 3;    % Prediction horizon
Duration = 5; % Simulation horizon

% For a usual control 0.1 , 5, 20

% Things specfic to the model starts here 

% Setting the initial state
x = [0.5;-0.3;-0.5;2];  % [0.5;0.3;0.4;]
xref1 = [0;0;0;0;];

% file = fopen('MPC_data','a');


% Things specfic to the model  ends here

options = optimoptions('fmincon','Algorithm','sqp','Display','none');
uopt = zeros(N,1);

u_max = 0;

% Apply the control input constraints
LB = -3*ones(N,1);
UB = 3*ones(N,1);

x_now = zeros(4,1);
x_next = zeros(4,1);
z = zeros(5,1);

x_now = x;

% Start simulation
fprintf('Simulation started.  It might take a while...\n')
xHistory = x;
steps = 100;
Duration = steps * Ts;
for ct = 1:(steps)
    
     ct

     u = NN_output(x_now,80,1,'neural_network_controller');
     
     z(1) = x_now(1) ;
     z(2) = x_now(2) ;
     z(3) = x_now(3) ;
     z(4) = x_now(4) ;

     z(5) = u ;
     
      
    x_next = system_eq_NN(x_now, Ts, u);
%     x_next = system_eq_dis(x_now, Ts, u);

    x = x_next;
    x_now = x_next;
% Save plant states for display.
    xHistory = [xHistory x]; 


%     Printing stuff

    disp('x(1) = '); x(1)
    disp('x(2) = '); x(2)
    disp('x(3) = '); x(3)

    disp('Control input = '); u
    
        
end

% fclose(file);

fprintf('Simulation finished!\n')

figure;
subplot(4,1,1);
plot(0:Ts:Duration,xHistory(1,:));
xlabel('time');
ylabel('x');

subplot(4,1,2);
plot(0:Ts:Duration,xHistory(2,:));
xlabel('time');
ylabel('y');

subplot(4,1,3);
plot(0:Ts:Duration,xHistory(3,:));
xlabel('time');
ylabel('z');

subplot(4,1,4);
plot(0:Ts:Duration,xHistory(4,:));
xlabel('time');
ylabel('z');
