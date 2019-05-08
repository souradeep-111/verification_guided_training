#ifndef run_python_h
#define run_python_h

#include <string.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <stdio.h>

using namespace std;

void create_a_fully_connected_network(
  vector< int > NN_config,
  string filename
);

// Some function which runs python script according to the points received,
// trains a neural network from it according to some MPC setting
void train_neural_network_controller(
  vector< vector< double > > input_points,
  vector< unsigned int > important_outputs,
  vector< double > scale_vector_input,
  vector< double > offset_adjustment,
  vector< vector< double > > control_bounds,
  char * controller_filename,
  char * system_model_filename,
  int no_of_steps,
  double learning_rate,
  int episodes,
  int batch_size
);

#endif
