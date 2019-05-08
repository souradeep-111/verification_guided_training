

Ts = 0.2;  % Sample Time
N = 3;    % Prediction horizon

% For a usual control 5 , 10, 6

% Things specfic to the model starts here 

% Setting the range of states
x_range = zeros(4,2);
x_range(1,1) = -1; x_range(1,2) = 1; % Cart position
x_range(2,1) = -10; x_range(2,2) = 10; % Cart speed
x_range(3,1) = -pi; x_range(3,2) = pi; % Angle
% x_range(4,1) = -50; x_range(4,2) = 50; % Angular Speed
x_range(4,1) = -20; x_range(4,2) = 20; % Angular Speed


% Generating the points for which we will train the neural network 
% for

no_of_data_points = zeros(4,1);
no_of_data_points(1) = 10;
no_of_data_points(2) = 10;
no_of_data_points(3) = 10;
no_of_data_points(4) = 10;

% Generating the data for the MPC Controller : 

options = optimoptions('fmincon','Algorithm','sqp','Display','none');
uopt = zeros(N,1);

% Apply the control input constraints
LB = -80*ones(N,1);
UB = 80*ones(N,1);

% Start simulation
fprintf('Generating the MPC data...  It might take a while...\n')


x = zeros(4,1);
xref1 = [0;0;0;0];
xref2 = [5;0;0;0];

xref = xref1;

a = 0;
b = 0;
c = 0;
d = 0;
% Saving data to a file

file = fopen('MPC_data','w');

% s = no_of_data_points^4; % no of samples
% fprintf(file,'%d\n',s);

s = 4 ; % dimension

for i = 1:(no_of_data_points(1) * no_of_data_points(2) * no_of_data_points(3) * no_of_data_points(4))
    x(1) = x_range(1,1) + ((x_range(1,2) - x_range(1,1)) * a)/no_of_data_points(1);
    x(2) = x_range(2,1) + ((x_range(2,2) - x_range(2,1)) * b)/no_of_data_points(2);
    x(3) = x_range(3,1) + ((x_range(3,2) - x_range(3,1)) * c)/no_of_data_points(3);
    x(4) = x_range(4,1) + ((x_range(4,2) - x_range(4,1)) * d)/no_of_data_points(4);
    
%     disp('New data set : ');
%     x(1)
%     x(2)
%     x(3)
%     x(4)
    
    
    a= a + 1;
    
    if(a == no_of_data_points(1))
        a = 0;
        b = b+1;
    end
    if(b == no_of_data_points(2))
        b = 0;
        c = c+1;
    end
    if(c == no_of_data_points(3))
        c = 0;
        d = d+1;
    end
    % A simple cost function for the MPC has been implemented here 
    % according to the reference input for the plant
    COSTFUN = @(u) objective_fcn(u,x,Ts,N,xref,uopt(1));
    
    % A simple contraint function which limits the angles and distance
    % travelled by the pendulum
    CONSFUN = @(u) constraint_fcn_for_pendulum(u,x,Ts,N);
    
    % taking the result of the optimization function implemented
    uopt = fmincon(COSTFUN,uopt,[],[],[],[],LB,UB,CONSFUN,options);

    % Take the first optimal control move.    
    z = uopt(1);
    
    i
    
    s = x(1);
    fprintf(file,'%d\n',s);
    
    s = x(2);
    fprintf(file,'%d\n',s);

    s = x(3);
    fprintf(file,'%d\n',s);

    s = x(4);
    fprintf(file,'%d\n',s);

    s = z;
    fprintf(file,'%d\n',s);
    
%     if(~(mod(i,10)))
%         i
%     end
end

fprintf('Data generation finished!\n')
fclose(file);
