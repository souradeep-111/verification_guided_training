#ifndef counter_examples_h
#define counter_examples_h

#include <iostream>
#include "propagate_intervals.h"

// some function which takes in the following inputs :
// 1. A closed loop neural network
// 2. Target set description
// 3. Violation steps count
//
// Algorithm :
// randomly sample the step before the last to find violations
// then, randomly sample the 2 steps before the last, and so on until there are no violations

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
);

// Returns true if the point is outside the region to violate
bool check_for_violation(
  vector< double > point ,
  vector< vector< double > > region_to_violate
);

#endif
