#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include <misc.hpp>

using namespace cl::sycl;

class convolver {
  public:
    // Constructor for convolution operations
    convolver(std::vector<Volume> weights_vector, short stride, float bias) : 
	    weights_vector{weights_vector}, stride{stride}, bias{bias} {
      filter_number = weights_vector.size();
      // Filter size; must be the same for all filters
      size = weights_vector[0].get_range().get(0);
      // Compute padding
      padding = floor(size/2);
    }
    // Constructor for pooling operations
    convolver(short size, short stride) : size{size}, stride{stride} {
      padding = floor(size/2);
    };
    int filter_number;
    short size;
    short stride;
    short padding;
    float bias;
    size_t input_width;
    size_t input_height;
    size_t depth;
    std::vector<Volume> weights_vector;
    Volume input_volume;

    Volume convolve(Volume &input_volume);
    Volume pool(Volume &input_volume);

    Volume pad(Volume &input_volume, short padding);

  
  private:
    Volume padded_volume;
    size_t padded_width;
    size_t padded_height;
    void pad();

};

class filter_functor{
  
  public:
    filter_functor(Volume weights_volume, short stride, float bias) : 
	    weights_volume{weights_volume}, stride{stride}, bias{bias} {
      size = weights_volume.get_range().get(0);
      depth = weights_volume.get_range().get(2);
    }
    size_t size;
    size_t depth;
    short stride;
    float bias;

    void operator() (Volume &input, Volume &output, short iteration);
  
  private:
    Volume weights_volume;

};

#endif
