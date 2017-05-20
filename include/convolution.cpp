#include <convolution.hpp>

using namespace cl::sycl;

Volume rand_input_generator(size_t width, size_t height, size_t depth){
  
  queue q;

  Volume v;

  buffer<float, 3> v_buf(&v, range<3>(width,height,depth));

  {
    q.submit( [&] (handler &cmdgroup) {
      auto V = v_buf.get_access<access::mode::write>(cmdgroup);
      cmdgroup.parallel_for<class rand_init>( range<3>(width,height,depth),
          	    [=] (id<3> index) {
          	    V[index]=rand()%256;
      });
    });
  }

  return v;
}

//conv_functor::conv_functor(volume<T> weights) : weights{weights} {}
