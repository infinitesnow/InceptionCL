#include <convolution.hpp>

using namespace cl::sycl;

Volume convolver::convolve(std::vector<Volume> weights, size_t size,short stride,short padding,int filter_number){
  
  pad(padding);

  Volume output_volume(range<3>(input_width,input_height,filter_number)); 
  
  for (int f=0;f<filter_number;f++){
    Volume filter_output = convolve_filter(weights[f],size,stride,padding);
    std::cout << "Filter output:" << std::endl;
    print_volume(filter_output);
  }
  
  //print_volume(output_volume);

  return output_volume;
};

inline Volume convolver::convolve_filter(Volume &weights_volume,short size,short stride,short padding) {
    queue q;
    Volume out(range<3>(input_width,input_height,1));
    
    q.submit( [&](handler &cmdgroup) {
      auto padded_a = padded_volume.get_access<access::mode::read>(cmdgroup);
      auto weights_a = weights_volume.get_access<access::mode::read>(cmdgroup);
      auto output_a = out.get_access<access::mode::write>(cmdgroup);
      cmdgroup.parallel_for<class convolve>( range<3>(input_width,input_height,1),
		    [=] (id<3> index) {
		    //output_a[index]=padded_a[index+id<3>(padding,padding,0)]*weights_a[index%id<3>(size,size,depth)];
		    output_a[index]=weights_a[index%id<3>(size,size,1)];
      });
    });

    return out;
};

void convolver::pad(short padding){
  padded_width = input_width+2*padding;
  padded_height = input_height+2*padding;

  std::cout << "Padding volume" << std::endl;

  clock_t time_a = clock();

  padded_volume = Volume( range<3>(padded_width,padded_height,depth) );
  
  initialize_volume(padded_volume,0);

  {

    queue q;

    q.submit( [&](handler &cmdgroup) {
      auto input_a = input_volume.get_access<access::mode::read>(cmdgroup);
      auto padded_a = padded_volume.get_access<access::mode::write>(cmdgroup);
      cmdgroup.parallel_for<class refill>( range<3>(input_width,input_height,depth),
		    [=] (id<3> index) {
		    padded_a[index+id<3>(padding,padding,0)]=input_a[index];
      });
    });
  }

  clock_t time_b = clock();

  std::cout << "Operation completed in " << time_b-time_a << " ticks." << std::endl;

  print_volume(padded_volume);

};

