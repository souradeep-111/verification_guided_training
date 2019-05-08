import tensorflow as tf
import numpy as np
import random
import sys

debug = True
my_random_seed = 125

name_of_controller_file = sys.argv[1]
name_of_system_model_file = sys.argv[2]
training_points_filename = sys.argv[3]
no_of_steps = int(sys.argv[4])

name_of_control_bounds_filename = sys.argv[5]
name_of_scaling_factor_filename = sys.argv[6]
name_of_bias_factor_filename = sys.argv[7]

learning_rate = float(sys.argv[8])
episodes = int(sys.argv[9])
batch_size = int(sys.argv[10])


# Reading control action ranges
control_bounds = []
data_stash = np.loadtxt(name_of_control_bounds_filename)
pointer = 0
no_of_controls = data_stash[pointer]
pointer = pointer + 1
for i in range(int(no_of_controls)):
    limits = []
    limits.append(data_stash[pointer])
    pointer = pointer + 1
    limits.append(data_stash[pointer])
    pointer = pointer + 1
    control_bounds.append(limits)

# Reading scaling factor
data_stash = np.loadtxt(name_of_scaling_factor_filename)

pointer = 0
no_of_states = int(data_stash[pointer])
pointer = pointer + 1

scaling_factor = []
for i in range(no_of_states):
    scaling_factor.append(float(data_stash[pointer]))
    pointer = pointer + 1

# Reading bias vector
data_stash = np.loadtxt(name_of_bias_factor_filename)
pointer = 0
no_of_dim = int(data_stash[pointer])
pointer = pointer + 1
bias_vector = []
for i in range(no_of_dim):
    bias_vector.append(float(data_stash[pointer]))
    pointer = pointer + 1


print("Learning rate = " , learning_rate)
print("Episodes = ", episodes)
print("batch size = ", batch_size)
# learning_rate = 0.001  #  2 --- 1e-3 // 3 --- 1e-4 // 5 - 1e-3 // 6 - 1e-4 // 7 -- 1e-3
# episodes = 40000 # 1 --- 1000 // 2 --- 1000 // 3 --- 1000 // 5 ---- 40000 // 6--- 20000 // 7 --- 20000
# batch_size = 1000 # 1 -- 20 // 2--- 10 // 3 ---- 20 // 4 --- 20 // 5 --- 50 // 6--- 1000 // 7 -- 500
# lower_bound_control = 0.0
# upper_bound_control = 8.0


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1, seed=my_random_seed)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def declare_tensor_for_upper_and_lower_bounds(bounds):

    # The lower bounds
    lb_name = "lower_bound_vector"
    python_scripts = []
    string = lb_name + " = " + "tf.constant(["
    for i in range(len(bounds)):
        if(i < (len(bounds) - 1)):
            string += str(bounds[i][0]) + ","
        else:
            string += str(bounds[i][0])

    string += "])"
    python_scripts.append(string)

    # The upper bounds
    ub_name = "upper_bound_vector"
    string = ub_name + " = " + "tf.constant(["
    for i in range(len(bounds)):
        if(i < (len(bounds) - 1)):
            string += str(bounds[i][1]) + ","
        else:
            string += str(bounds[i][1])

    string += "])"
    python_scripts.append(string)

    return lb_name, ub_name, python_scripts

def declare_a_tensor_vector(array, name):

    python_scripts = []
    string = name + " = " + "tf.constant(["
    for i in range(len(array)):
        if(i < (len(array) - 1)):
            string += str(array[i]) + ","
        else:
            string += str(array[i])

    string += "])"
    python_scripts.append(string)
    return python_scripts



def read_and_get_the_structure_of_network(file_name):
    global debug
    no_of_state_vars = 0
    no_of_control_vars = 0
    control_network_hidden_config = []

    data_stash = np.loadtxt(file_name)
    pointer = 0
    no_of_state_vars = int(data_stash[pointer])
    pointer = pointer + 1
    no_of_control_vars = int(data_stash[pointer])
    pointer = pointer + 1
    no_of_hidden_layers = int(data_stash[pointer])


    pointer = pointer + 1
    control_network_hidden_config = []
    for i in range(no_of_hidden_layers):
        control_network_hidden_config.append(int(data_stash[pointer]))
        pointer = pointer + 1

    if(debug):
        print("For filename = ", file_name)
        print("No of hidden layers = ", no_of_hidden_layers)
        print("No of input vars = ", no_of_state_vars)
        print("No of output vars = ", no_of_control_vars)
        print("Hidden layer configi = ", control_network_hidden_config)
    return no_of_state_vars, no_of_control_vars, control_network_hidden_config


# This function assumes that when the scripts are run "W_x" is defined
def create_the_lines_to_compute_the_control_output(names_weights_and_biases, input_name, output_name, index):
    python_scripts = []
    no_of_computations_needed = len(names_weights_and_biases) / 2
    python_line = "r_0" + "_" + str(index) + " = " + input_name
    python_scripts.append(python_line)
    for i in range(no_of_computations_needed):
        python_line = "r_" + str(i+1) + "_" + str(index) + " = tf.nn.relu(tf.matmul(r_" + str(i) + "_" + str(index) + ",W_" + \
        str(i) + ") + b_" + str(i) + ")"
        python_scripts.append(python_line)
    python_line = output_name + " = r_" + str(no_of_computations_needed) + "_" + str(index)
    python_scripts.append(python_line)

    return python_scripts

# This function assumes that when the scripts are run "Ws_x" is defined
def create_the_lines_to_compute_the_system_output(names_weights_and_biases, input_name, output_name, index):
    python_scripts = []
    no_of_computations_needed = len(names_weights_and_biases) / 2
    python_line = "q_0" + "_" + str(index) + " = " + input_name
    python_scripts.append(python_line)
    for i in range(no_of_computations_needed):
        python_line = "q_" + str(i+1)+ "_" + str(index) + " = tf.nn.relu(tf.matmul(q_" + str(i) + "_" + str(index) + ",Ws_" + \
        str(i) + ") + bs_" + str(i) + ")"
        python_scripts.append(python_line)
    python_line = output_name + " = q_" + str(no_of_computations_needed) + "_" + str(index)
    python_scripts.append(python_line)

    return python_scripts

def combine_the_control_network_and_system_network(names_of_system_network, names_of_control_network, input_name, output_name, times):
    global bias_vector
    global scaling_factor
    global control_bounds

    if(debug):
        print("Bias vector globally available = ", bias_vector)
        print("Scaling vector = ", scaling_factor)
        print("control bounds = ", control_bounds)

    python_scripts = []

    string = "interm_state_space_input_0 = " + input_name
    python_scripts.append(string)

    control_lb_name, control_ub_name, scripts = declare_tensor_for_upper_and_lower_bounds(control_bounds)
    python_scripts += scripts

    state_var_scaling_name = "scaling_factor"
    state_var_shifting_name = "bias_var_name"

    scripts = declare_a_tensor_vector(scaling_factor, state_var_scaling_name)
    python_scripts += scripts

    scripts = declare_a_tensor_vector(bias_vector, state_var_shifting_name)
    python_scripts += scripts

    # The control shifting variable
    control_shift = []
    total_range = 0
    for i in range(len(control_bounds)):
        total_range = (control_bounds[i][1] - control_bounds[i][0])
        control_shift.append((total_range)/2.0)
    scalar_ctrl_bnd = total_range

    control_shift_name = "control_output_shift"
    scripts = declare_a_tensor_vector(control_shift, control_shift_name)
    python_scripts += scripts

    for i in range(times):
        intermediate_control_output = "interm_control_input_" + str(i)
        state_space_input = "interm_state_space_input_" + str(i)

        lines_to_comp_control_out = create_the_lines_to_compute_the_control_output(names_of_control_network, state_space_input,\
        intermediate_control_output, i)
        python_scripts += lines_to_comp_control_out

        if(i > 0):
            string = state_space_input + "_a" + " = tf.add( " + state_space_input + " , " + state_var_shifting_name + ")"
            python_scripts.append(string)
            string = state_space_input+ "_b" + " = tf.multiply( " + state_space_input + "_a" + " , " + state_var_scaling_name + " ) "
            python_scripts.append(string)
        else:
            string = state_space_input + "_b" + " = " + state_space_input
            python_scripts.append(string)

        string = intermediate_control_output + "_a" + " = tf.clip_by_value( " + intermediate_control_output + " , " + str(0.0)\
        + " , " + str(scalar_ctrl_bnd) + " ) "
        python_scripts.append(string)

        string = intermediate_control_output + "_b"+ " = tf.subtract( " + intermediate_control_output + "_a"+ ", " + control_shift_name + " ) "
        python_scripts.append(string)


        intermediate_sys_input = "interm_concat_input_" + str(i)
        string = intermediate_sys_input + "_a" + " = tf.concat( [" + state_space_input+ "_b" + ", " + intermediate_control_output + "_b"+ "], 1)"
        python_scripts.append(string)

        state_space_output = "interm_state_space_output_" + str(i)
        lines_to_comp_system_output = create_the_lines_to_compute_the_system_output(names_of_system_network, \
        intermediate_sys_input+ "_a", state_space_output, i)
        python_scripts += lines_to_comp_system_output


        string = "interm_state_space_input_" + str(i+1) + " = " + state_space_output
        python_scripts.append(string)


    string = state_space_output+ "_a" + " = tf.add( " + "interm_state_space_input_" + str(times) + " , " + state_var_shifting_name + ")"
    python_scripts.append(string)

    string = output_name + " = tf.multiply( " + state_space_output + "_a" + " , " + state_var_scaling_name + " ) "
    python_scripts.append(string)

    return python_scripts

# Read the control network's structure, size of hidden layers etc
# and produce a string of variable names

no_of_state_vars, no_of_control_vars, control_net_structure = \
read_and_get_the_structure_of_network(name_of_controller_file)

c_network_weight_names = []
c_network_bias_names = []

for i in range(len(control_net_structure) + 1):
    c_network_weight_names.append( "W_"+str(i) )
    c_network_bias_names.append( "b_" + str(i) )



# Create a function which generates lines to declare weights and bias variables,
# using the names received and initializes them with the neural network controller
# file

data_stash = np.loadtxt(name_of_controller_file)
pointer = 0
no_of_inputs = int(data_stash[pointer])
pointer = pointer + 1
no_of_outputs = int(data_stash[pointer])
pointer = pointer + 1
no_of_hidden_layers = int(data_stash[pointer])
pointer = pointer + 1

control_network_config = []
control_network_config.append(no_of_inputs) # The input layer size
for i in range(no_of_hidden_layers):
    control_network_config.append(int(data_stash[pointer]))
    pointer = pointer + 1
control_network_config.append(no_of_outputs)

for j in range(1,len(control_network_config)):
    weight_matrix = []
    bias_var = []
    no_of_rows = control_network_config[j]
    no_of_cols = control_network_config[j-1]
    for row_no in range(no_of_rows):
        weight_vector = []
        for col_no in range(no_of_cols):
            data = float(data_stash[pointer])
            pointer = pointer + 1
            weight_vector.append(data)

        weight_matrix.append(weight_vector)

        data = float(data_stash[pointer])
        pointer = pointer + 1
        bias_var.append(data)

    weight_matrix = np.array(weight_matrix, dtype = np.float32)
    weight_matrix = np.transpose(weight_matrix)
    exec("W_%d = tf.Variable(weight_matrix)" %(j-1) )
    exec("b_%d = tf.Variable(bias_var)" %(j-1) )
    # exec("W_%d = weight_variable( [len(weight_matrix), len(weight_matrix[0])] ) " %(j-1) )
    # exec("b_%d = bias_variable([len(bias_var)]) " %(j-1) )
    # exec("W_%d = weight_variable( [no_of_cols, no_of_rows] ) " %(j-1) )
    # exec("b_%d = bias_variable([no_of_rows]) " %(j-1) )

#
# Create a function which when given the string having variables for the control network
# generate the python lines for taking the input and computing the output. Using the names
# for the inputs and output names received.

# Read the system network's structure and return names for the variables
no_of_input_sys_network, no_of_outputs_sys_network, system_network_structure = \
read_and_get_the_structure_of_network(name_of_system_model_file)

s_network_weight_names = []
s_network_bias_names = []

for i in range(len(system_network_structure) + 1):
    s_network_weight_names.append( "Ws_"+str(i) )
    s_network_bias_names.append( "bs_" + str(i) )

# Create a function which does the following : Reads in the weights
# and bias matrix and then assigns it to the  tf variables as constants
data_stash = np.loadtxt(name_of_system_model_file)
pointer = 0
no_of_system_inputs = int(data_stash[pointer])
pointer = pointer + 1
no_of_system_outputs = int(data_stash[pointer])
pointer = pointer + 1
no_of_hidden_layers = int(data_stash[pointer])
pointer = pointer + 1

system_network_config = []
system_network_config.append(no_of_system_inputs) # The input layer size
for i in range(no_of_hidden_layers):
    system_network_config.append(int(data_stash[pointer]))
    pointer = pointer + 1
system_network_config.append(no_of_system_outputs)

if(debug):
    print("System network config = ", system_network_config)

for j in range(1,len(system_network_config)):
    weight_matrix = []
    bias_var = []
    no_of_rows = system_network_config[j]
    no_of_cols = system_network_config[j-1]
    for row_no in range(no_of_rows):
        weight_vector = []
        for col_no in range(no_of_cols):
            data = float(data_stash[pointer])
            pointer = pointer + 1
            weight_vector.append(data)
        weight_matrix.append(weight_vector)

        data = float(data_stash[pointer])
        pointer = pointer + 1
        bias_var.append(data)

    weight_matrix = np.array(weight_matrix, dtype = np.float32)
    weight_matrix = np.transpose(weight_matrix)
    exec("Ws_%d = tf.constant(weight_matrix, shape = [len(weight_matrix), len(weight_matrix[0])] , dtype = tf.float32)" %(j-1)  )
    exec("bs_%d = tf.constant(bias_var, shape = [len(bias_var)] , dtype = tf.float32)" %(j-1)  )


# A function which generates the lines for the following :
# Given the Names of the control network pieces, and the Names of the system network variables
# Write lines which is going to combine the two and then repeat them for a given number of steps
# Have the first input and the last output given names accordingly

# Train the network according to the set of points received as inputs, and the cost function
# and save the neural network controller

training_points = np.loadtxt(training_points_filename)

pointer = 0
input_dimension = int(training_points[pointer])
pointer = pointer + 1
no_of_points = int(training_points[pointer])
pointer = pointer + 1

# print("No of points training from  = ", no_of_points)

x_data = np.zeros((no_of_points, input_dimension)).astype(np.float32)
y_data = np.zeros((no_of_points, input_dimension)).astype(np.float32)

for i in range(no_of_points):
    for j in range(input_dimension):
        x_data[i][j] = float(training_points[pointer])
        pointer = pointer + 1

np.random.shuffle(x_data)

x = tf.placeholder('float', shape = [None, input_dimension])
y_ = tf.placeholder('float', shape = [None, input_dimension])

input_name = "x"
output_name = "y"

python_script = combine_the_control_network_and_system_network(s_network_weight_names + s_network_bias_names,\
  c_network_weight_names + c_network_bias_names, input_name, output_name, no_of_steps)

for i in range(len(python_script)):
    if(debug):
        print(python_script[i])
    exec(python_script[i])


exec("mean_square_error = tf.reduce_mean(tf.square(" + output_name + "- y_)) " )
training = tf.train.AdamOptimizer(learning_rate).minimize(mean_square_error)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

min_error = np.inf
max_error = -np.inf
min_error = np.inf

# print(" before training W_1 = ", sess.run(W_1))
for j in range(1,len(control_network_config)):
    no_of_rows = control_network_config[j]
    no_of_cols = control_network_config[j-1]
    exec("W_store_%d = sess.run(W_%d) " %(j-1, j-1) )
    exec("b_store_%d = sess.run(b_%d) " %(j-1, j-1) )


for _ in range(episodes):
    total_error = 0
    for i in range(no_of_points - batch_size + 1):
        _, error  = sess.run([training, mean_square_error  ] , \
        feed_dict={x: x_data[i:i+batch_size,:], y_:y_data[i:i+batch_size,:]})
        total_error += error
        # print("Error = ", error)
        # print("interm_state_space_output_3  = ", out)
        # sys.exit()
        if (total_error > max_error):
            max_error = total_error
    if(max_error < min_error):
        min_error = max_error
        for j in range(1,len(control_network_config)):
            no_of_rows = control_network_config[j]
            no_of_cols = control_network_config[j-1]
            exec("W_store_%d = sess.run(W_%d) " %(j-1, j-1) )
            exec("b_store_%d = sess.run(b_%d) " %(j-1, j-1) )

        print("Min error = ",min_error)
    max_error = -np.inf


# print(" after training W_1 = ", sess.run(W_1))


file = open(name_of_controller_file , "w" )
file.writelines(str(no_of_inputs) + '\n')
file.writelines(str(no_of_outputs) + '\n')
file.writelines(str(no_of_hidden_layers) + '\n')

for i in range(no_of_hidden_layers):
    file.writelines(str(control_network_config[1+i]) + '\n')


for j in range(1,len(control_network_config)):
    weight_matrix = []
    # exec("weight_matrix =  sess.run(W_" + str(j-1) + ")")
    # exec("bias_var =  sess.run(b_" + str(j-1) + ")")
    exec("weight_matrix =  (W_store_" + str(j-1) + ")")
    exec("bias_var =  (b_store_" + str(j-1) + ")")

    # Some kind of transpose needs to happen here
    no_of_rows = control_network_config[j]
    no_of_cols = control_network_config[j-1]

    for row_no in range(no_of_rows):
        for col_no in range(no_of_cols):
            file.writelines(str(weight_matrix[col_no, row_no]) + '\n')

        # bias_vector.append(data)
        file.writelines(str(bias_var[row_no]) + '\n')


file.close()
