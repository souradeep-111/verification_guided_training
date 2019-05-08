#include "run_verification.h"

// Some function which takes in a property and neural network and
// tries to verify it

// inputs : The closed loop system network
//        : The reach set to be proved
//        : The number of steps that you need to search for

// Algorithm :
// Do reach set propagation
// check if the property is being satisfied : i.e propagate by N steps, and check if all of them are inside the target set by then.
// if yes : You are done .
// if not : ask for counter examples and return

bool debug_run_verif = true;

bool run_verification(
  network_handler system_network,
  vector< vector< datatype > > starting_region,
  vector< vector< datatype > > target_region,
  int no_of_steps,
  int acc_in_reach_sets,
  vector< unsigned int > important_outputs,
  vector< double > scale_vector_input,
  vector< double > offset_already,
  vector< vector< double > > total_directions,
  vector< vector< double > >& counter_examples
)
{


  set_info current_set, next_set;
  current_set.region_constr = starting_region;
  current_set.time_stamp = 0;

  queue < set_info > reach_sets;
  reach_sets.push(current_set);

  vector< set_info > all_reach_sets;
  map< int, set_info > ordered_reach_sets;

  vector< vector< double > > biases_next(total_directions.size(), vector< double >(2,0));


  for(int i = 0; i < no_of_steps; i++)
  {
    current_set = reach_sets.front();
    reach_sets.pop();

    simulate_accelerated(system_network, acc_in_reach_sets, important_outputs,
                         scale_vector_input, offset_already, current_set.region_constr, biases_next);


    adjust_offset(biases_next, offset_already);
    scale_vector(biases_next, scale_vector_input);

    convert_direction_biases_to_constraints(total_directions, biases_next, next_set.region_constr);

    cout << "At time = " << current_set.time_stamp << " reach set boundaries are " << endl;
    print_biases(biases_next);

    next_set.time_stamp = current_set.time_stamp + acc_in_reach_sets;
    reach_sets.push(next_set);

    if(find_if_subset(next_set.region_constr, target_region))
    {
      break;
    }

    ordered_reach_sets.insert(pair<int, set_info > (i+1, next_set));
  }


  if(find_if_subset(next_set.region_constr, target_region))
  {
    return true;
  }
  else
  {
    counter_examples.clear();
    vector< double > counter_example;
    for(int j = 0; j < 2  ; j++)
    {
      find_counter_example( j, system_network, important_outputs, /*ordered_reach_sets[j+1].region_constr*/starting_region,
        target_region, no_of_steps, offset_already, scale_vector_input, counter_example);
      counter_examples.push_back(counter_example);
    }
    if(debug_run_verif)
    {
      cout << "Done with counter examples " << endl;
      cout << "No of counter examples are " << counter_examples.size() << endl;
      print_2D_vector(counter_examples);
      // exit(0);

    }
    return false;
  }
  return false;

}
