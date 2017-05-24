#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include <misc.hpp>

using namespace cl::sycl;

class convolver {
  public:
    convolver(std::vector<Volume> weights, short stride, short padding) : weights{weights}, stride{stride}, padding{padding} {
      filter_number = weights.size();
    }
    int filter_number;
    short padding;
    short stride;
    size_t input_width;
    size_t input_height;
    size_t depth;
    std::vector<Volume> weights;
    Volume input_volume;

    Volume operator() (Volume &input_volume);

  
  private:
    Volume padded_volume;
    size_t padded_width;
    size_t padded_height;
    void pad(short padding);

};

class filter_functor{
  
  public:
    filter_functor(Volume weights, short stride) : weights{weights}, stride{stride} {
      size = weights.get_range().get(0);
      depth = weights.get_range().get(2);
    }
    size_t size;
    size_t depth;
    short stride;

    Volume operator() (Volume &input);
  
  private:
    Volume weights;

};

class maxpool_functor{
  size_t size;
  maxpool_functor(size_t size) : size{size} {}
  Volume operator() (Volume);
};

#endif
