#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include <misc.hpp>

class filter{
  
  public:
    filter();
    filter(Volume weights_volume, short stride, float bias) : 
	    weights_volume{weights_volume}, stride{stride}, bias{bias} {
      size = weights_volume.get_range().get(0);
      depth = weights_volume.get_range().get(2);
      BOOST_LOG_TRIVIAL(trace) << "Building a filter of size " << size << ", depth " << depth;
    }
    size_t size;
    size_t depth;
    short stride;
    float bias;

    void operator() (Volume& input, Volume& output, short f);
  
  private:
    Volume weights_volume;

};

class convolver {
  public:
	  //
    // Constructor for convolution operations
    convolver(std::vector<Volume> weights_vector, short stride, float bias) : 
	   stride{stride}, bias{bias} {
      // Compute padding
      this->padding = floor(size/2);

      // Get number of filters
      this->filter_number = weights_vector.size();
      // Filter size; must be the same for all filters
      this->size = weights_vector[0].get_range().get(0);
      // Instantiate filters
      for (short f=0;f<filter_number;f++){
        BOOST_LOG_TRIVIAL(trace) << "Convolver: Instantiating filter " << f+1;
	filter ft=filter(weights_vector[f],stride,bias);
        this->filters_vector.push_back(ft);
      };

    };

    // Constructor for pooling operations
    convolver(short size, short stride) : size{size}, stride{stride} {
      this->padding = floor(size/2);
    };

    short size;
    short stride;
    short padding;
    float bias;

    void initialize(Volume &v, cl::sycl::queue q){
      this->input_volume = v;
      this->q= q;
      this->input_width = input_volume.get_range().get(0);
      this->input_height = input_volume.get_range().get(1);
      this->input_depth = input_volume.get_range().get(2);
      output_volume = Volume(cl::sycl::range<3>(input_width,input_height,input_depth));
      pad();
    };
    Volume input_volume;
    size_t input_width;
    size_t input_height;
    size_t input_depth;
    Volume output_volume;

    cl::sycl::queue q;

    Volume* convolve();
    Volume* pool();

  private:
    Volume padded_volume;
    size_t padded_width;
    size_t padded_height;
    size_t padded_depth;
    void pad();

    int filter_number;
    std::vector<filter> filters_vector;
};

#endif
