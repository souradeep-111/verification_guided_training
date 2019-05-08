function final_val = system_eq_dis(x_initial,time, control_input)

function dxdt = cont(t,x)
    dxdt =[ x(3) - x(2);
            x(3);
            control_input;];
end

[t ,y] = ode45(@cont, [0 time],x_initial);
s = size(y);
final_val = y(s(1),:);
final_val = final_val';

end