#include <CL/sycl.hpp>
#include <convolution.hpp>

const size_t WIDTH = 32;
const size_t HEIGHT = 32;
const size_t DEPTH = 3;

int main(){
  Volume input_volume = rand_input_generator(WIDTH,HEIGHT,DEPTH);
  print_volume(input_volume,WIDTH,HEIGHT,DEPTH);
  Volume v33 = convolve(input_volume,3,1,1);
  print_volume(v33,34,34,3);
  return 0;
};
