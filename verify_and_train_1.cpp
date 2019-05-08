#include "./src/propagate_intervals.h"
#include "./src/run_python.h"
#include "./src/run_verification.h"

bool debug_main = true;

int main(int argc, char ** argv)
{
    sherlock_parameters.gradient_rate = 1e-6;
    sherlock_parameters.grad_scaling_factor = 2e2;

    sherlock_parameters.MILP_M = 1e10;
    sherlock_parameters.MILP_tolerance = 1e-2 ;
    sherlock_parameters.MILP_e_tolerance =  1e-6;
    sherlock_parameters.scale_factor_for_M = 1.0;

    auto state_space_dim = 2;
    auto control_dim = 1;
    auto hidden_nodes = 25;
    auto no_of_initial_points = 50;
    auto no_of_steps = 30;
    auto no_of_unrollings = 10;
    auto acc_in_reach_sets = 1;
    auto no_of_extra_directions = 2;
    bool only_simulate = false;
    int invariancy_steps = 5;
    double learning_rate = 1e-4;
    int episodes = 1000;
    int batch_size = 20;

    // Total trainings = 2

    vector< vector< double > > state_interval(state_space_dim, vector< double >(2));

    state_interval[0][0] = -1.0;state_interval[0][1] = 1.0;
    state_interval[1][0] = -1.0;state_interval[1][1] = 1.0;


  // Declare a neural network
    // create some random neural network using a python script
    // and store it in a file with the given name

  string controller_network_filename("./network_files/control_ex_12_2d/controller_2.nt");

  vector< int > network_configuration ; /* This includes even the input and output */
  network_configuration.push_back(state_space_dim);
  network_configuration.push_back(hidden_nodes);
  network_configuration.push_back(control_dim);

  if(!only_simulate)
  {
    create_a_fully_connected_network(network_configuration, controller_network_filename);
  }

  // Get some initial system points and train with that
  vector< vector< double > > initial_set_of_points(no_of_initial_points);
  // the state space that I expect the control to work in



  vector< vector< double > > state_space_constraint;
  create_constraint_from_interval(state_space_constraint, state_interval);
  auto seed = 0;
  for(auto & each_point : initial_set_of_points)
  {
    seed+= 13;
    return_random_sample_with_seed(state_space_constraint, each_point, seed);
  }
  // Try to run verification if passes it's good.
  /* NOTE: Need to try multiple initial sets, but as of now just trying 1 */

  // THIS SECTION TRIES TO PREPARE THE DIFFERENT ARGUMENTS NEEDED FOR RUNNING THE
  // VERIFICATION


  char state_vars_file_1[] = "./network_files/control_ex_12_2d/neural_network_information_1" ;
  char state_vars_file_2[] = "./network_files/control_ex_12_2d/neural_network_information_2" ;

  network_handler system_1(state_vars_file_1);
  network_handler system_2(state_vars_file_2);

  vector< vector< vector< datatype > > > w_1;
  vector< vector< datatype > > b_1;
  vector< vector< vector< datatype > > > w_2;
  vector< vector< datatype > > b_2;
  vector< vector< vector< datatype > > > w_4;
  vector< vector< datatype > > b_4;


  system_1.return_network_information(w_1, b_1);
  system_2.return_network_information(w_2, b_2);
  w_4 = w_1;
  b_4 = b_1;
  patch_networks_vertically(w_4,b_4,w_2,b_2);

  char name[] = "./network_files//control_ex_12_2d/state_vars";
  write_network_to_file(w_4, b_4, name);

  char state_vars_file[] = "./network_files/control_ex_12_2d/state_vars" ;

  // char controller_file[] = "./network_files/control_ex_12_2d/controller.nt" ;
  char * controller_file = new char[controller_network_filename.length()+1];
  strcpy(controller_file, controller_network_filename.c_str());

  char closed_loop_file[] = "./network_files/control_ex_12_2d/composed_network_non_linear";

  merge_networks(-4, 1, closed_loop_file, state_vars_file, controller_file);


  network_handler system_network(closed_loop_file);
  unsigned extra_directions = no_of_extra_directions;

  vector< vector < datatype > > directions(extra_directions, vector< datatype >(2));
  directions[0][0] = 1;directions[0][1] = 1;
  directions[1][0] = 1;directions[1][1] = -1;


  vector< vector< vector< datatype > > > weights;
  vector< vector< datatype > > biases;
  system_network.return_network_information(weights, biases);


  vector< datatype > offset_already(state_space_dim);
  offset_already[0] = -80;
  offset_already[1] = -80;

  vector< vector< datatype > > control_bounds(control_dim, vector< datatype >(2));
  control_bounds[0][0] = -4;
  control_bounds[0][1] = 4;


  add_directions_to_output(weights, biases, directions,
    sherlock_parameters.constr_comb_offset, offset_already);

  system_network.update_information(weights, biases);



  system_network.return_network_information(weights, biases);

  vector< unsigned int > important_outputs(state_space_dim + extra_directions, 1);
  fill(important_outputs.begin() + state_space_dim, important_outputs.end(), 0);

  vector< datatype > scaling_factor(state_space_dim + extra_directions, 0.1);



  vector< vector< datatype > > total_directions(state_space_dim + extra_directions, vector< datatype >(2) );
  total_directions[0][0] = 1;total_directions[0][1] = 0;
  total_directions[1][0] = 0;total_directions[1][1] = 1;

  unsigned int i, j , k, steps;
  i = 0;
  while(i < directions.size())
  {
    total_directions[i + state_space_dim] = directions[i];
    i++;
  }

  vector< datatype > scale_vector_input(state_space_dim + extra_directions, 0.1);

  vector< datatype > scale_vector_output(state_space_dim + extra_directions, 10);

  vector< vector< datatype > > biases_now(state_space_dim + extra_directions, vector< datatype >(2));
  biases_now[0][0] = 0.4;biases_now[0][1] = 0.45;
  biases_now[1][0] = -0.55;biases_now[1][1] = -0.5;

  // For the invariant proving , steps : 2
  // biases_now[0][0] = -0.02;biases_now[0][1] = 0.02;
  // biases_now[1][0] = -0.02;biases_now[1][1] = 0.02;


  biases_now[2][0] = -10;biases_now[2][1] = 10;
  biases_now[3][0] = -10;biases_now[3][1] = 10;


  // vector< datatype > offset_amount(2 + extra_directions);
  // offset_amount[0] = -80;
  // offset_amount[1] = -80;

  offset_already.resize(state_space_dim + no_of_extra_directions);
  fill(offset_already.begin() + no_of_extra_directions, offset_already.end(), sherlock_parameters.constr_comb_offset);

  vector< vector< datatype > > biases_next(state_space_dim + extra_directions, vector< datatype >(2,0));
  vector< vector< datatype > > region_constraints;

  convert_direction_biases_to_constraints(total_directions,
                                          biases_now,
                                          region_constraints);



  set_info current_set, next_set, target_set;
  current_set.region_constr = region_constraints;
  current_set.time_stamp = 0;

  vector< vector< datatype > > target_region;
  vector< datatype > constraint(2 + 1);

  // x_0 < 0.02 ... [ -x_0 + 0 x_1 + 0.02 > 0 ]
  constraint[0] = -1; constraint[1] = 0; constraint[2] = 0.02;
  target_region.push_back(constraint);

  // x_0 > -0.02 ... [ x_0 + 0 x_1 + 0.02 > 0 ]
  constraint[0] = 1; constraint[1] = 0; constraint[2] = 0.02;
  target_region.push_back(constraint);

  // x_1 < 0.02 ... [ 0 x_0 + (-1) x_1 + 0.02 > 0 ]
  constraint[0] = 0; constraint[1] = -1; constraint[2] = 0.02;
  target_region.push_back(constraint);

  // x_1 > -0.02 ... [ 0 x_0 + x_1 + 0.02 > 0 ]
  constraint[0] = 0; constraint[1] = 1;constraint[2] = 0.02;
  target_region.push_back(constraint);

  //


  target_set.region_constr = target_region;

  vector< double > offset_for_network;
  offset_for_network.resize(state_space_dim);
  i = 0;
  while(i < state_space_dim)
  {
    offset_for_network[i] = offset_already[i];
    i++;
  }

  if(!only_simulate)
  {
    train_neural_network_controller(initial_set_of_points, important_outputs, scale_vector_input, offset_for_network,
      control_bounds, controller_file, state_vars_file, no_of_unrollings, learning_rate, episodes, batch_size);
  }

    merge_networks(-4, 1, closed_loop_file, state_vars_file, controller_file);


    network_handler current_network_1(closed_loop_file);
    vector< vector< vector< datatype > > > buffer_weights;
    vector< vector< datatype > > buffer_biases;

    current_network_1.return_network_information(buffer_weights, buffer_biases);

    add_directions_to_output(buffer_weights, buffer_biases, directions,
      sherlock_parameters.constr_comb_offset, offset_already);

    system_network.update_information(buffer_weights, buffer_biases);


  if(debug_main)
  {
    cout << "First set of training completed successfully." << endl;
  }

  int retraining_counts = 0;

  vector< vector< double > > counter_examples_collection;
  if(!only_simulate)
  {
    while( /* until verification passes */
      !run_verification(
        system_network, region_constraints, target_region,
        no_of_steps, acc_in_reach_sets, important_outputs,
        scale_vector_input, offset_already, total_directions, counter_examples_collection
      )
    )
    {

      assert(!counter_examples_collection.empty());
      initial_set_of_points.insert(initial_set_of_points.end(),
      counter_examples_collection.begin(), counter_examples_collection.end());

      train_neural_network_controller(initial_set_of_points, important_outputs, scale_vector_input, offset_for_network,
        control_bounds, controller_file, state_vars_file, no_of_unrollings, learning_rate, episodes, batch_size);

        merge_networks(-4, 1, closed_loop_file, state_vars_file, controller_file);


        network_handler current_network_1(closed_loop_file);
        vector< vector< vector< datatype > > > buffer_weights;
        vector< vector< datatype > > buffer_biases;

        current_network_1.return_network_information(buffer_weights, buffer_biases);

        add_directions_to_output(buffer_weights, buffer_biases, directions,
          sherlock_parameters.constr_comb_offset, offset_already);

        system_network.update_information(buffer_weights, buffer_biases);

      retraining_counts++;
      cout << "Currently at retraining count number = " << retraining_counts << endl;
    }

    cout << " --------------- Controller design done and verified ------------------ " << endl;

  }

  cout << "System Simulation Starts " << endl;
  cout << " ------------------------------------------------------------ " << endl;
  simulate_system_and_create_plots(no_of_steps, system_network, acc_in_reach_sets, important_outputs,
    scale_vector_input, offset_already, current_set, target_set, total_directions );


  cout << "All simulation and plotting ends -------------- " << endl;
  cout << "Starting to prove the invariant set condition " << endl;

  biases_now[0][0] = -0.02;biases_now[0][1] = 0.02;
  biases_now[1][0] = -0.02;biases_now[1][1] = 0.02;
  biases_now[2][0] = -10;biases_now[2][1] = 10;
  biases_now[3][0] = -10;biases_now[3][1] = 10;



  convert_direction_biases_to_constraints(total_directions,
                                          biases_now,
                                          region_constraints);


  current_set.region_constr = region_constraints;
  current_set.time_stamp = 0;


  bool res = prove_invariant_set(system_network, invariancy_steps, important_outputs, scale_vector_input, offset_already,
    current_set, total_directions, biases_next);

  if(res)
  {
    cout << "Invariancy proved, with number of steps = "  << invariancy_steps  << endl;
    cout << " Final set biase computed  : " << endl;
    print_biases(biases_next);
  }
  else
  {
    cout << invariancy_steps << " not enough for proving invariancy " << endl;
  }

  cout << " Thanks for your patience, but anyway, what choice did you have ? ;) " << endl;



  return 0;
}
