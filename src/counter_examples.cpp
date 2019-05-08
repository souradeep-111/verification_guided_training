#include "counter_examples.h"

bool debug_counter_ex = false;
int max_counter_ex = 10;

typedef vector< vector < vector < double > > > weight_collection;
typedef vector < vector < double > > bias_collection;
typedef vector< vector < vector < vector< double > > > > vector_of_weight_collection;
typedef vector < vector < vector< double > > > vector_of_bias_collection;
int no_of_random_samples = 1e4;

void find_counter_example(
  int seed,
  network_handler system_network,
  vector< unsigned int > important_outputs,
  vector< vector< double > > initial_set,
  vector< vector < double > > target_set,
  int steps_to_violation,
  vector< double > bias,
  vector< double > scaling_factors,
  vector< double >& counter_example
)
{

  int no_of_inputs = system_network.no_of_inputs;
  assert(scaling_factors.size() == bias.size());
  assert(!initial_set.empty());
  assert(initial_set[0].size() == (no_of_inputs + 1) );
  assert(!target_set.empty());
  assert(target_set[0].size() == (no_of_inputs + 1) );
  assert(steps_to_violation > 0);



  vector_of_weight_collection all_weights(no_of_inputs);
  vector_of_bias_collection all_biases(no_of_inputs);

  vector< vector< unsigned int > > active_weights;

  vector< double > x(no_of_inputs);
  vector< double > x_next(no_of_inputs);

  weight_collection weights_buffer;
  bias_collection bias_buffer;
  for(auto i = 0; i < no_of_inputs; i++)
  {
    system_network.return_network_information(weights_buffer, bias_buffer);
    system_network.cast_to_single_output_network(weights_buffer, bias_buffer, i+1);
    all_weights[i] = weights_buffer;
    all_biases[i] = bias_buffer;
  }

  vector< double > scale_vector_input(no_of_inputs);
  vector< double > scale_vector_output(no_of_inputs);
  for(int i = 0; i < no_of_inputs; i++)
  {
    scale_vector_input[i] = 1.0/scaling_factors[i];
    scale_vector_output[i] = scaling_factors[i];
  }

  vector< double > potential_counter_example;
  vector< vector< double > > counter_example_repo;

  int ex_count = 0;
  for(int i = 0; i < (int)no_of_random_samples; i++)
  {
      // pick a random sample with 'i' as the seed
      return_random_sample_with_seed(initial_set, potential_counter_example, i*23 + seed);

      auto x = potential_counter_example;
      auto x_next = potential_counter_example;

      // propagate that point through the system networks
      for(auto step_count = 0; step_count < steps_to_violation; step_count ++)
      {
        for(auto dim = 0; dim < no_of_inputs; dim++)
        {
          if(important_outputs[dim])  // THIS PART IS A BIT BRITTLE, BE CAREFUL. YOU ARE ASSUMING AN ORDERING
          {
            x_next[dim] = compute_network_output(x, all_weights[dim], all_biases[dim], active_weights);
            x_next[dim] -= bias[dim];
            x_next[dim] /= scaling_factors[dim];

          }
        }
        x = x_next;
      }
      // Check if it is actually a counter example
      // if yes, add it to the repo
      // else continue

      if(check_for_violation(x_next, target_set))
      {
        counter_example = potential_counter_example;
        if(debug_counter_ex)
        {
          cout << "Found a concrete counter example " << endl;
        }
        return ;
      }
  }

  if(debug_counter_ex)
  {
    cout << "No counter example found using random search " << endl;
  }
  return ;


}

bool check_for_violation(
  vector< double > point ,
  vector< vector< double > > region_to_violate
)
{

  vector< vector< vector < double > > > neg_constraints;
  if(check_counter_example(region_to_violate, neg_constraints, point))
  {
    return false;
  }
  else
  {
    return true;
  }

}
