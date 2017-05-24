#include <convolution.hpp>

using namespace cl::sycl;

Volume convolver::convolve(std::vector<Volume> weights, size_t size,short stride,short padding,int filter_number){
  
  pad(padding);

  Volume output_volume(range<3>(input_width,input_height,filter_number)); 
  
  for (int f=0;f<filter_number;f++){
    
    Volume weights_volume = weights[f];
    filter_functor filter_functor(weights_volume);
    Volume filter_output(range<3>(size,size,depth));
    Volume output(range<3>(input_width,input_height,1));

    queue q;
    
    q.submit( [&](handler &cmdgroup) {
      auto padded_a = padded_volume.get_access<access::mode::read>(cmdgroup);
      auto weights_a = weights_volume.get_access<access::mode::read>(cmdgroup);
      auto filter_output_a = filter_output.get_access<access::mode::write>(cmdgroup);
      auto output_a = output.get_access<access::mode::write>(cmdgroup);
      
      cmdgroup.parallel_for<class convolve>( range<3>(input_width,input_height,1), [=] (id<3> base_index) {
		    id<3> output_index=base_index;
                    id<3> input_index=base_index+id<3>(padding,padding,0);
                    id<3> index = id<3>(0,0,0);
		    float result=0;
		    float current_product=0;
                    while ( index[2] < depth ){
                      while ( index[1] < size ){
                        while ( index[0] < size ){
		          current_product = padded_a[input_index+index-id<3>(1,1,0)]*weights_a[index];
		          filter_output_a[index]=current_product;
			  result+=current_product;
                          index[0]+=stride;
		        };
		        index[0]=0;
		        index[1]+=stride;
		      }; 
		      index[1]=0;
		      index[2]+=stride;
		    };
		    output_a[output_index]=result;
             });
    });

    std::cout << "Last kernel output:" << std::endl;
    print_volume(filter_output);

    std::cout << "Output filter " << f+1 << ":" << std::endl;
    print_volume(output);
  
  };

  return output_volume;
};

inline void convolver::print_subvolume(item<3> base_it, auto buffer_a){
}

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

