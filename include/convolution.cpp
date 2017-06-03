#include <convolution.hpp>

using namespace cl::sycl;

Volume convolver::convolve(Volume &v) {

  input_volume = v;
  input_width = input_volume.get_range().get(0);
  input_height = input_volume.get_range().get(1);
  depth = input_volume.get_range().get(2);

  BOOST_LOG_TRIVIAL(info) << "Convolver: Convolving (" << size << "x" << size  << ") volume of size " 
	  << volume_size(input_volume) << "...";

  pad();

  queue q;
  Volume output(range<3>(input_width,input_height,filter_number));

  for (short f=0;f<filter_number;f++){
    
    BOOST_LOG_TRIVIAL(trace) << "Convolver: Instantiating filter " << f+1;
    Volume* weights_volume = &weights_vector[f];
    filter ft(*weights_volume,stride,bias);

    q.submit( [&](handler &filter_cmdgroup) {
      BOOST_LOG_TRIVIAL(debug) << "Convolver: Submitting filter " << f+1;
      filter_cmdgroup.single_task( [=](){
		      ft(padded_volume,output,f)
		      });
    });

  };

  BOOST_LOG_TRIVIAL(debug) << "Convolver: Convolution output (" << size << "x" << size  << ") volume size: " 
	  << volume_size(output);
  
  //std::this_thread::sleep_for(std::chrono::seconds(1));
  
  return output;
};

void convolver::pad(){
  padded_width = input_width+2*padding;
  padded_height = input_height+2*padding;

  BOOST_LOG_TRIVIAL(debug) << "Convolver: Padding volume... " << volume_size(input_volume);

  padded_volume = Volume( range<3>(padded_width,padded_height,depth) );
  BOOST_LOG_TRIVIAL(debug) << "Convolver: Initializing padding volume " << volume_size(padded_volume)<< " to 0";
  initialize_volume(padded_volume,0);

  queue q;

  q.submit( [&](handler &cmdgroup) {
    BOOST_LOG_TRIVIAL(trace) << "Convolver: Submitting padding task to queue ("
   	 << volume_size(padded_volume) << ")";
    auto input_a = input_volume.get_access<access::mode::read>(cmdgroup);
    auto padded_a = padded_volume.get_access<access::mode::write>(cmdgroup);
    cmdgroup.parallel_for<class refill>( range<3>(input_width,input_height,depth),
      	    [=] (id<3> index) {
            BOOST_LOG_TRIVIAL(trace) << "Convolver: reinserting element " << index_tostring(index)
	    	<< " into padding volume " << volume_size(padded_volume);
      	    padded_a[index+id<3>(padding,padding,0)]=input_a[index];
    });
  });

  //print_volume(padded_volume);
};

void filter::operator() (Volume& input, Volume& output, short f) {
  BOOST_LOG_TRIVIAL(trace) << "Filter " << f+1 << " ("<<size<<"x"<<size<<"x"<<depth<<"): starting convolution";
  size_t input_width = input.get_range().get(0);
  size_t input_height = input.get_range().get(1);
  size_t depth = input.get_range().get(2);

  size_t output_width = output.get_range().get(0);
  size_t output_height = output.get_range().get(0);

  short padding = floor(size/2);

  Volume filter_output = Volume(range<3>(size,size,depth));

  queue q;
  
  q.submit( [&](handler &cmdgroup) {
    BOOST_LOG_TRIVIAL(trace) << "Filter " << f+1 
    	<< " ("<<size<<"x"<<size<<"x"<<depth<<")" 
    	<< ": Submitting task to queue";
    auto input_a = input.get_access<access::mode::read>(cmdgroup);
    auto weights_a = weights_volume.get_access<access::mode::read>(cmdgroup);
    auto filter_output_a = filter_output.get_access<access::mode::write>(cmdgroup);
    auto output_a = output.get_access<access::mode::write>(cmdgroup);
    
    cmdgroup.parallel_for<class convolve>( range<3>(output_width,output_height,1), [=] (id<3> base_index) {
            BOOST_LOG_TRIVIAL(trace) << "Filter " << f+1 
	    	<< " ("<<size<<"x"<<size<<"x"<<depth<<")" 
	    	<< ": entering parallel for"; 
	    /* 
	    Input and output spaces have the same dimensions. We are iterating over this space, 
	    then we calculate the input index, which has an offset equal to the padding. 
	    Then, we iterate around the input index.
	    */
	    id<3> offset = id<3>(padding,padding,0);
      	    id<3> input_index=base_index+offset;
	    // We write on the ith level of the output volume
      	    id<3> output_index=base_index+id<3>(0,0,f);
            id<3> index = id<3>(0,0,0);
            BOOST_LOG_TRIVIAL(trace) << "Filter " << f+1 
	    	<< " ("<<size<<"x"<<size<<"x"<<depth<<")" 
	    	<< ": Parallel for with base index " << index_tostring(base_index) 
	    	<< ", offset " << index_tostring(offset) 
	    	<< ", output index " << index_tostring(output_index); 
            float result=0;
	    float current_input_value=0;
	    float current_weight=0;
            float current_product=0;
            while ( index[2] < depth ){
              while ( index[1] < size ){
                while ( index[0] < size ){
	          current_input_value = input_a[input_index+index-offset];	
	          current_weight = weights_a[index];
                  current_product = current_input_value*current_weight;
	          filter_output_a[index]=current_product;
                  result+=current_product;
	          BOOST_LOG_TRIVIAL(trace) << "Filter " << f+1 
	                  << ": Iteration with index " << index_tostring(index)
	                  << ", calculating " 
	                  << current_input_value << "*" << current_weight << "=" << current_product; 
                  index[0]+=1;
                };
                index[0]=0;
                index[1]+=1;
              }; 
              index[1]=0;
              index[2]+=1;
            };
            result+=bias;
    	    output_a[output_index]=result;
	    BOOST_LOG_TRIVIAL(trace) << "Filter " << f+1 
		    << ": Result ("<<index_tostring(output_index)<<"): " << result;
           });
  });
  
  //std::cout << "Last kernel output volume for filter " << f+1 << ":" << std::endl; 
  //print_volume(filter_output);

};

Volume convolver::pool(Volume &v){
  BOOST_LOG_TRIVIAL(info) << "POOL: Pooling...";

  input_volume = v;
  input_width = input_volume.get_range().get(0);
  input_height = input_volume.get_range().get(1);
  depth = input_volume.get_range().get(2);

  size_t output_width = input_width;
  size_t output_height = input_height;

  pad();

  queue q;
  Volume output(range<3>(input_width,input_height,depth));

  q.submit( [&](handler &cmdgroup) {
    BOOST_LOG_TRIVIAL(trace) << "POOL: submitting task to queue";
    auto input_a = padded_volume.get_access<access::mode::read>(cmdgroup);
    auto output_a = output.get_access<access::mode::write>(cmdgroup);
    
    cmdgroup.parallel_for<class pool>( range<3>(output_width,output_height,depth), [=] (id<3> base_index) {
		id<3> offset = id<3>(padding,padding,0);
      	        id<3> input_index=base_index+offset;
      	        id<3> output_index=base_index;
                id<3> index = id<3>(0,0,0);
                BOOST_LOG_TRIVIAL(trace) << "POOL: Parallel for with base index " << index_tostring(base_index); 
		float max = -1;
                while ( index[1] < size ){
                  while ( index[0] < size ){
                    max=std::max(input_a[input_index+index-offset],max);
		    BOOST_LOG_TRIVIAL(trace) << "POOL: max is now set to " << max; 
                    index[0]+=1;
                  };
                  index[0]=0;
                  index[1]+=1;
                }; 
	    BOOST_LOG_TRIVIAL(trace) << "POOL: max for this tile is " << max;
    	    output_a[output_index]=max;
           });
  });

  BOOST_LOG_TRIVIAL(debug) << "Pooling output size: " << volume_size(output) << std::endl;

  return output;
  
};
