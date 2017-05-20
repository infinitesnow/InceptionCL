#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include <misc.hpp>

class conv_functor{
  
  public:
    conv_functor(Volume weights,size_t size) : weights{weights},size(size) {}
    size_t size;
    Volume operator() (Volume);
  
  private:
    Volume weights;

};

class maxpool_functor{
  size_t size;
  maxpool_functor(size_t size);
  Volume operator() (Volume);
};

Volume convolve(Volume &v,size_t size,short stride,short padding);

#endif
