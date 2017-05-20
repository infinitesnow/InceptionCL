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

class convolver {
  public:
    size_t input_width;
    size_t input_height;
    size_t depth;
    size_t padded_width;
    size_t padded_height;
    Volume input_volume;
    Volume padded_volume;
    convolver(Volume &input_volume) : input_volume{input_volume} {
      input_width = input_volume.get_range().get(0);
      input_height =  input_volume.get_range().get(1);
      depth =  input_volume.get_range().get(2);
    }

    Volume convolve(size_t size,short stride,short padding);
    void pad(short padding);

    template <size_t s>
    inline Volume generate_stub_weights(size_t depth){
      std::cout << "Generating stub weights" << std::endl;
      //Volume w = rand_volume_generator(s,s,depth);
      Volume w = cl::sycl::buffer<float,3>( cl::sycl::range<3>(s,s,depth));      
      initialize_volume(w,1);
      print_volume(w);
    };
};

#endif
