#include <CL/sycl.hpp>
#include <convolution.hpp>


int main(){

  // Define input parameters
  size_t input_width = 32;
  size_t input_height = 32;
  size_t input_depth = 3;
  
  // Create a random input volume
  Volume input_volume = rand_volume_generator(input_width,input_height,input_depth);
  print_volume(input_volume);
  
  // Create a convolver object
  convolver c(input_volume);
  
  // Weights are generated by a stub method
  std::vector<Volume> weights;
  int filter_number = 4;
  for (int i=1; i<=filter_number; i++){
    std::cout << "Generating stub weights for filter " << i+1 << std::endl;
    Volume w =generate_stub_weights(3,input_depth,float(i));
    weights.push_back(w);
  }
  
  c.convolve(weights,3,1,1,filter_number);
  std::cout << "Finished" << std::endl;
  return 0;
};
