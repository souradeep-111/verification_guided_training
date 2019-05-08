#ifndef run_verification_h
#define run_verification_h

#include <iostream>
#include <vector>
#include <queue>
#include <map>
#include "propagate_intervals.h"
#include "network_computation.h"
#include "counter_examples.h"

using namespace std;

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
);

#endif
