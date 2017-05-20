#include <CL/sycl.hpp>
#include <convolution.hpp>

const size_t WIDTH = 32;
const size_t HEIGHT = 32;
const size_t DEPTH = 3;

int main(){
  Volume input_volume = rand_volume_generator(WIDTH,HEIGHT,DEPTH);
  print_volume(input_volume);
  convolver c(input_volume);
  c.convolve(3,1,1);
  return 0;
};
