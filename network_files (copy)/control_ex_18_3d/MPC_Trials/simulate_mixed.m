Ts = 0.2;  % Sample Time
N = 3;    % Prediction horizon
Duration = 20; % Simulation horizon


% For a usual control 0.5 , 5, 10

% Things specfic to the model starts here 

% Setting the initial state
x = [ 0.4;3;3;5];
xref1 = [0;0;0;0;];

% file = fopen('MPC_data','a');


% Things specfic to the model  ends here

options = optimoptions('fmincon','Algorithm','sqp','Display','none');
uopt = zeros(N,1);

u_max = 0;

% Apply the control input constraints
LB = -80*ones(N,1);
UB = 80*ones(N,1);

x_now  = zeros(4,1);
x_next = zeros(4,1);
z = zeros(4,1);


% Start simulation
fprintf('Simulation started.  It might take a while...\n')
xHistory = x;
steps = Duration/Ts;

for ct = 1:(steps)
    ct
    xref = xref1;
    
    % A simple cost function for the MPC has been implemented here 
    % according to the reference input for the plant
    COSTFUN = @(u) objective_fcn(u,x,Ts,N,xref,uopt(1));
    
    % A simple contraint function which limits the angles and distance
    % travelled by the pendulum
    CONSFUN = @(u) constraint_fcn(u,x,Ts,N);
    
    % taking the result of the optimization function implemented
    uopt = fmincon(COSTFUN,uopt,[],[],[],[],LB,UB,CONSFUN,options);

%     uopt(1) = NN_output(x, 100, 0.1, 'neural_network_information_controller');

    % Implement first optimal control move and update plant states.
%     x = system_eq_dis(x, Ts, uopt(1));
    x = system_eq_NN(x, Ts, uopt(1));

% Save plant states for display.
    xHistory = [xHistory x]; 


%     Printing stuff

    disp('x(1) = '); x(1)
    disp('x(2) = '); x(2)
    disp('x(3) = '); x(3)
    disp('x(4) = '); x(4)


    disp('Control input = '); uopt(1)
    
        
end

% fclose(file);

fprintf('Simulation finished!\n')

figure;
subplot(4,1,1);
plot(0:Ts:Duration,xHistory(1,:));
xlabel('time');
ylabel('Cart Pos');

subplot(4,1,2);
plot(0:Ts:Duration,xHistory(2,:));
xlabel('time');
ylabel('Cart Velocity');

subplot(4,1,3);
plot(0:Ts:Duration,xHistory(3,:));
xlabel('time');
ylabel('Angle');

subplot(4,1,4);
plot(0:Ts:Duration,xHistory(4,:));
xlabel('time');
ylabel('Angular speed');
