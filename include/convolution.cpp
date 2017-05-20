#include <convolution.hpp>

using namespace cl::sycl;

Volume convolve(Volume &input_volume,size_t size,short stride,short padding){
  size_t input_width = input_volume.get_range().get(0);
  size_t input_height =  input_volume.get_range().get(1);
  size_t depth =  input_volume.get_range().get(2);
  size_t padded_width = input_width+2*padding;
  size_t padded_height = input_height+2*padding;
 
  queue q;

  std::cout << "Padding volume" << std::endl;

  clock_t time_a = clock();

  Volume padded_volume( range<3>(padded_width,padded_height,depth) );

  {
    q.submit( [&] (handler &cmdgroup) {
      auto input_a = input_volume.get_access<access::mode::read>(cmdgroup);
      auto padded_a = padded_volume.get_access<access::mode::write>(cmdgroup);
      cmdgroup.parallel_for<class pad>( range<3>(input_width,input_height,depth),
          	    [=] (id<3> index) {
		    ;
      });
    });
  }

  clock_t time_b = clock();

  std::cout << "Operation completed in " << time_b-time_a << " ticks." << std::endl;

};
