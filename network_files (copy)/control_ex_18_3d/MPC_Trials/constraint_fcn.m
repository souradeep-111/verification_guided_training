function [c, ceq]  = constraint_fcn(u,x,Ts,N)

x_current = x;

mark = 1;

for i = 1:N
    
%     x_next = system_eq_dis(x_current, Ts,u(i));
    x_next = system_eq_NN(x_current, Ts,u(i));

    if(( x_next(1) > -1) && (x_next(1) < 1) && ...
        ( x_next(2) > -1) && (x_next(2) < 1) && ...
        ( x_next(3) > -1) && (x_next(3) < 1)...      
        )
        mark = mark * 1;
    else
        mark = 0;
    end
    
    x_current = x_next;
    
end

if(mark)
    c = -1;
else
    c = 1;
end
ceq = [];
   

end