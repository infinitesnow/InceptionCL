#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include <CL/sycl.hpp>

using Volume = std::vector<std::vector<std::vector<float>>>;

class conv_functor{
  
  public:
    conv_functor(Volume weights){}
    Volume operator() (Volume);
  
  private:
    Volume weights;

};

class maxpool_functor{
  size_t size;
  maxpool_functor(size_t size);
  Volume operator() (Volume);
};

Volume rand_input_generator(size_t width, size_t height, size_t layers);
#endif
