#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include <misc.hpp>

class conv_functor{
  
  public:
    conv_functor(Volume weights) : weights{weights} {
      size = weights.get_range().get(0);
    }
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

class convolver {
  public:
    size_t input_width;
    size_t input_height;
    size_t depth;
    Volume input_volume;
    Volume padded_volume;
    convolver(Volume &input_volume) : input_volume{input_volume} {
      input_width = input_volume.get_range().get(0);
      input_height =  input_volume.get_range().get(1);
      depth =  input_volume.get_range().get(2);
    }

    Volume convolve(std::vector<Volume> weights, size_t size,short stride,short padding,int filter_number);
  
  private:
    size_t padded_width;
    size_t padded_height;
    void pad(short padding);
    inline Volume convolve_filter(Volume &weights_volume,short size,short stride,short padding);

};

#endif
