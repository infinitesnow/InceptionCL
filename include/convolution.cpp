#include <convolution.hpp>

using namespace cl::sycl;

Volume convolver::convolve(Volume &input_volume) {
  std::cout << "Convolving (" << size << "x" << size  << ")..." << std::endl;

  this->input_volume = input_volume;
  input_width = input_volume.get_range().get(0);
  input_height = input_volume.get_range().get(1);
  depth = input_volume.get_range().get(2);

  pad();

  queue q;
  Volume output(range<3>(input_width,input_height,filter_number));

  for (short f=0;f<filter_number;f++){
    
    q.submit( [&](handler &filter_cmdgroup) {
      Volume weights_volume = weights_vector[f];
      filter_functor filter_functor(weights_volume,stride);
      filter_functor(padded_volume,output,f);
    });

  };

  std::cout << "Convolution output (" << size << "x" << size  << ") volume:" << std::endl;
  print_volume(output);
  
  return output;
};

void convolver::pad(){
  padded_width = input_width+2*padding;
  padded_height = input_height+2*padding;

  std::cout << "Padding volume..." << std::endl;

  padded_volume = Volume( range<3>(padded_width,padded_height,depth) );
  initialize_volume(padded_volume,0);

  queue q;

  q.submit( [&](handler &cmdgroup) {
    auto input_a = input_volume.get_access<access::mode::read>(cmdgroup);
    auto padded_a = padded_volume.get_access<access::mode::write>(cmdgroup);
    cmdgroup.parallel_for<class refill>( range<3>(input_width,input_height,depth),
      	    [=] (id<3> index) {
      	    padded_a[index+id<3>(padding,padding,0)]=input_a[index];
    });
  });

  //print_volume(padded_volume);
};

void filter_functor::operator() (Volume &input, Volume &output, short f) {
  size_t input_width = input.get_range().get(0);
  size_t input_height = input.get_range().get(1);
  size_t depth = input.get_range().get(2);

  size_t output_width = output.get_range().get(0);
  size_t output_height = output.get_range().get(0);

  short padding = floor(size/2);

  Volume filter_output = Volume(range<3>(size,size,depth));

  queue q;
  
  q.submit( [&](handler &cmdgroup) {
    auto input_a = input.get_access<access::mode::read>(cmdgroup);
    auto weights_a = weights_volume.get_access<access::mode::read>(cmdgroup);
    auto filter_output_a = filter_output.get_access<access::mode::write>(cmdgroup);
    auto output_a = output.get_access<access::mode::write>(cmdgroup);
    
    cmdgroup.parallel_for<class convolve>( range<3>(output_width,output_height,1), [=] (id<3> base_index) {
		// Input and output spaces have the same dimensions. We are iterating over this space, then we calculate
		// the input index, which has an offset equal to the padding. Then, we iterate around the input index. 
		id<3> offset = id<3>(padding,padding,0);
      	        id<3> input_index=base_index+offset;
		// We write on the ith level of the output volume
      	        id<3> output_index=base_index+id<3>(0,0,f);
                id<3> index = id<3>(0,0,0);
                float result=0;
                float current_product=0;
                while ( index[2] < depth ){
                  while ( index[1] < size ){
                    while ( index[0] < size ){
                      current_product = input_a[input_index+index-offset]*weights_a[index];
		      filter_output_a[index]=current_product;
                      result+=current_product;
                      index[0]+=1;
                    };
                    index[0]=0;
                    index[1]+=1;
                  }; 
                  index[1]=0;
                  index[2]+=1;
                };
    	    output_a[output_index]=result;
           });
  });
  
  std::cout << "Last kernel output volume for filter " << f+1 << ":" << std::endl; 
  print_volume(filter_output);

};

Volume convolver::pool(Volume &input_volume){
  std::cout << "Pooling..." << std::endl;

  this->input_volume = input_volume;
  input_width = input_volume.get_range().get(0);
  input_height = input_volume.get_range().get(1);
  depth = input_volume.get_range().get(2);

  size_t output_width = input_width;
  size_t output_height = input_height;

  pad();

  queue q;
  Volume output(range<3>(input_width,input_height,depth));

  q.submit( [&](handler &cmdgroup) {
    auto input_a = padded_volume.get_access<access::mode::read>(cmdgroup);
    auto output_a = output.get_access<access::mode::write>(cmdgroup);
    
    cmdgroup.parallel_for<class pool>( range<3>(output_width,output_height,depth), [=] (id<3> base_index) {
                //std::cout << "hi" << base_index[0]<<base_index[1] << base_index[2] << std::endl;
		id<3> offset = id<3>(padding,padding,0);
      	        id<3> input_index=base_index+offset;
      	        id<3> output_index=base_index;
                id<3> index = id<3>(0,0,0);
		std::vector<float> elements;
                while ( index[1] < size ){
                  while ( index[0] < size ){
                    elements.push_back(input_a[input_index+index-offset]);
		    index[0]+=1;
                  };
                  index[0]=0;
                  index[1]+=1;
                }; 
    	    output_a[output_index]=*std::max_element(std::begin(elements), std::end(elements));
           });
  });

  std::cout << "Pool output volume:" << std::endl;
  print_volume(output);

  return output;
  
};
