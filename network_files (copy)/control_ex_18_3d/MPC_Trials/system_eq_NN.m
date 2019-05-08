function final_val = system_eq_NN(x_initial,time, control_input)

final_val = zeros(3,1);

z = zeros(4,1);

z(1) = x_initial(1);
z(2) = x_initial(2);
z(3) = x_initial(3);
z(4) = control_input;

final_val = NN_output(z, 100, 1, 'neural_network.nt');
% final_val(1) = NN_output(z, 80, 0.1, 'neural_network_information_1');
% final_val(2) = NN_output(z, 200, 0.1, 'neural_network_information_2');
% final_val(3) = NN_output(z, 200, 0.1, 'neural_network_information_3');
% final_val(4) = NN_output(z, 250, 0.1, 'neural_network_information_4');

end