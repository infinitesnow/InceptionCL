#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include <misc.hpp>

using namespace cl::sycl;

class convolver {
  public:
    convolver(std::vector<Volume> weights_vector, short stride, short padding) : weights_vector{weights_vector}, stride{stride}, padding{padding} {
      filter_number = weights_vector.size();
    }
    int filter_number;
    short padding;
    short stride;
    size_t input_width;
    size_t input_height;
    size_t depth;
    std::vector<Volume> weights_vector;
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
    filter_functor(Volume weights_volume, short stride) : weights_volume{weights_volume}, stride{stride} {
      size = weights_volume.get_range().get(0);
      depth = weights_volume.get_range().get(2);
    }
    size_t size;
    size_t depth;
    short stride;

    void operator() (Volume &input, Volume &output, short iteration);
  
  private:
    Volume weights_volume;

};

class maxpool_functor{
  size_t size;
  maxpool_functor(size_t size) : size{size} {}
  Volume operator() (Volume);
};

#endif
