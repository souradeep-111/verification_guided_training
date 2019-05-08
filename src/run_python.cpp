#include "run_python.h"

bool debug_run_python = true;

string path = "./src/";
void create_a_fully_connected_network(vector< int > NN_config , string filename )
{
  string py_script_name = "./src/initialize_a_network.py ";
  filename = filename + " ";
  string system_call = "python " + py_script_name + filename;
  for(int i = 0; i < NN_config.size() ; i++)
  {
    system_call += (to_string(NN_config[i]) + " " );
  }

  if(debug_run_python)
  {
    cout << "System call being made = " << system_call << endl;
  }

  char * c_string = new char [system_call.length() + 1];
  strcpy(c_string, system_call.c_str());
  system(c_string);

}

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
)
{
   string sample_points_filename = path + "input_points_for_controller_training";

   char * c_filename = new char [sample_points_filename.length()+1];
   strcpy (c_filename, sample_points_filename.c_str());
   ofstream file;
   // The input file for the training points
   file.open(c_filename);

   auto dim = input_points[0].size();

   file << dim << "\n";
   file << input_points.size() << "\n";

   int i = 0;
   while(i < input_points.size())
   {
     int j = 0;
     while(j < dim)
     {
       file << input_points[i][j] << "\n";
       j++;
     }
     i++;
   }
   file.close();

   // The file for the scaling vectors
   string scaling_vectors_filename = path + "state_network_scaling_factors";

   c_filename = new char [scaling_vectors_filename.length()+1];
   strcpy (c_filename, scaling_vectors_filename.c_str());
   file.open(c_filename);

   dim = scale_vector_input.size();
   file << input_points[0].size() << "\n";
   i = 0;
   while(i < dim)
   {
     if(important_outputs[i])
     {
       file << scale_vector_input[i] << "\n";
     }
     i++;
   }
   file.close();

   // The file for the bias vectors
   string bias_vectors_filename = path + "state_network_bias_factors";

   c_filename = new char [bias_vectors_filename.length()+1];
   strcpy (c_filename, bias_vectors_filename.c_str());
   file.open(c_filename);

   dim = offset_adjustment.size();
   file << dim << "\n";
   i = 0;
   while(i < dim)
   {
     file << offset_adjustment[i] << "\n";
     i++;
   }
   file.close();

   // The file for the control bounds
   string control_bounds_filename = path + "control_vectors_bound";

   c_filename = new char [control_bounds_filename.length()+1];
   strcpy (c_filename, control_bounds_filename.c_str());
   file.open(c_filename);

   dim = control_bounds.size();
   file << dim << "\n";
   i = 0;
   while(i < dim)
   {
     file << control_bounds[i][0] << "\n";
     file << control_bounds[i][1] << "\n";
     i++;
   }
   file.close();


   string s_controller_filename(controller_filename);
   string s_system_model_filename(system_model_filename);
   string python_script_name = path + "train_MPC_network_controller.py ";
   string system_call = "python " + python_script_name + s_controller_filename + " "
   + s_system_model_filename + " " + sample_points_filename + " " + to_string(no_of_steps) + " " +
   control_bounds_filename + " " + scaling_vectors_filename + " " + bias_vectors_filename + " "
   + to_string(learning_rate) + " " + to_string(episodes) + " " + to_string(batch_size);

   if(debug_run_python)
   {
     cout << "System call being made : " << system_call << endl;
   }
   char * c_string = new char [system_call.length() + 1];
   strcpy(c_string, system_call.c_str());
   system(c_string);


}
