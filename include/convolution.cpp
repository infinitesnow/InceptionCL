#include <convolution.hpp>

using namespace cl::sycl;

void filter::operator() (Volume& input, Volume& output, short f, cl::sycl::queue& q) {

  BOOST_LOG_TRIVIAL(trace) << "Filter " << f+1 << " ("<<size<<"x"<<size<<"x"<<depth<<"): starting convolution";
  size_t input_width = input.get_range().get(0);
  size_t input_height = input.get_range().get(1);
  size_t depth = input.get_range().get(2);

  size_t output_width = output.get_range().get(0);
  size_t output_height = output.get_range().get(0);

  short padding = floor(size/2);

  Volume filter_output = Volume(range<3>(size,size,depth));

  q.submit( [&](handler &cmdgroup) {
    BOOST_LOG_TRIVIAL(trace) << "Filter " << f+1 
    	<< " ("<<size<<"x"<<size<<"x"<<depth<<")" 
    	<< ": Submitting task to queue";
    auto input_a = input.get_access<access::mode::read>(cmdgroup);
    auto weights_a = weights_volume.get_access<access::mode::read>(cmdgroup);
    auto filter_output_a = filter_output.get_access<access::mode::write>(cmdgroup);
    auto output_a = output.get_access<access::mode::write>(cmdgroup);
    
    cmdgroup.parallel_for<class convolve>( range<3>(output_width,output_height,1), [=] (id<3> base_index) {
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
    	                  << " ("<<size<<"x"<<size<<"x"<<depth<<")" 
		          << ", operating on volume " << volume_size(input)
			  << ", on base index " << index_tostring(input_index)
			  << ", offset" << index_tostring(offset)
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
	    BOOST_LOG_TRIVIAL(trace) << "Filter "<< f+1
    	            << " ("<<size<<"x"<<size<<"x"<<depth<<")" 
		    << ", operating on volume " << volume_size(input)
		    << ", on base index " << index_tostring(input_index)
		    << ": applying ReLU to "<< result+bias<<","<<0;
            result=std::max(result+bias,(float)0.0);
    	    output_a[output_index]=result;
	    BOOST_LOG_TRIVIAL(trace) << "Filter " << f+1 
    	            << " ("<<size<<"x"<<size<<"x"<<depth<<")" 
		    << ", operating on volume " << volume_size(input)
		    << ", on base index " << index_tostring(input_index)
		    << ": Result (to "<<index_tostring(output_index)<<"): " << result;
           });
  });
  
  //std::cout << "Last kernel output volume for filter " << f+1 << ":" << std::endl; 
  //print_volume(filter_output);

};

void convolver::pad_init(){
  this->padded_width = input_width+2*padding;
  this->padded_height = input_height+2*padding;
  this->padded_depth = input_depth;
  this->padded_volume = Volume( range<3>(padded_width,padded_height,padded_depth) );

  BOOST_LOG_TRIVIAL(debug) << "Convolver: Initializing padded volume to 0, from " 
	  << input_width<<"x"<<input_height<<"x"<<input_depth
	  << " to " 
	  << input_width<<"x"<<input_height<<"x"<<input_depth;

  initialize_volume(padded_volume,0,q);

};

void convolver::pad_fill(){
  BOOST_LOG_TRIVIAL(debug) << "Convolver: Filling padded volume";
  q.submit( [&](handler &cmdgroup) {
    BOOST_LOG_TRIVIAL(trace) << "Convolver: Submitting padding task to queue ("
   	 << volume_size(padded_volume) << ")";
    auto input_a = input_volume.get_access<access::mode::read>(cmdgroup);
    auto padded_a = padded_volume.get_access<access::mode::write>(cmdgroup);
    cmdgroup.parallel_for<class refill>( range<3>(input_width,input_height,input_depth),
      	    [=] (id<3> index) {
            BOOST_LOG_TRIVIAL(trace) << "Convolver: reinserting element " << input_a[index] << " " 
	    	<< index_tostring(index)
	    	<< " into padding volume " << volume_size(padded_volume);
      	    padded_a[index+id<3>(padding,padding,0)]=input_a[index];
    });
  });
}

Volume* convolver::convolve() {

  BOOST_LOG_TRIVIAL(info) << "Convolver: Convolving ("<<size<<"x"<<size<<") volume of size " 
	  << input_width<<"x"<<input_height<<"x"<<input_depth << "...";

  if (this->is_soft) input_volume = *input;

  pad_fill();

  for (short f=0;f<filter_number;f++){
    q.submit( [&](handler &filter_cmdgroup) {
      BOOST_LOG_TRIVIAL(debug) << "Convolver: Submitting filter " << f+1;
      filters_vector[f](padded_volume,output_volume,f,q);
    });
  };

  return &output_volume;
};


Volume* convolver::pool(){
  BOOST_LOG_TRIVIAL(info) << "POOL: Pooling...";

  if (this->is_soft) input_volume = *input;
  
  pad_fill();

  this->padded_volume = Volume( range<3>(padded_width,padded_height,padded_depth) );
  BOOST_LOG_TRIVIAL(debug) << "Convolver: Initializing padding volume " << volume_size(padded_volume)<< " to 0";
  initialize_volume(padded_volume,0,q);

  q.submit( [&](handler &cmdgroup) {
    BOOST_LOG_TRIVIAL(trace) << "Convolver: Submitting padding task to queue ("
   	 << volume_size(padded_volume) << ")";
    auto input_a = input_volume.get_access<access::mode::read>(cmdgroup);
    auto padded_a = padded_volume.get_access<access::mode::write>(cmdgroup);
    cmdgroup.parallel_for<class refill>( range<3>(input_width,input_height,input_depth),
      	    [=] (id<3> index) {
            BOOST_LOG_TRIVIAL(trace) << "Convolver: reinserting element " << input_a[index] << " " 
	    	<< index_tostring(index)
	    	<< " into padding volume " << volume_size(padded_volume);
      	    padded_a[index+id<3>(padding,padding,0)]=input_a[index];
    });
  });
};

Volume* convolver::convolve() {

  BOOST_LOG_TRIVIAL(info) << "Convolver: Convolving (" << size << "x" << size  << ") volume of size " 
	  << volume_size(input_volume) << "...";

  for (short f=0;f<filter_number-1;f++){
    q.submit( [&](handler &filter_cmdgroup) {
      BOOST_LOG_TRIVIAL(debug) << "Convolver: Submitting filter " << f+1;
      filters_vector[f](padded_volume,output_volume,f,q);
    });
  };

  q.submit( [&](handler &cmdgroup) {
    BOOST_LOG_TRIVIAL(trace) << "POOL: submitting task to queue";
    auto input_a = padded_volume.get_access<access::mode::read>(cmdgroup);
    auto output_a = output_volume.get_access<access::mode::write>(cmdgroup);
    
    cmdgroup.parallel_for<class pool>( range<3>(output_width,output_height,input_depth), [=] (id<3> base_index) {
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

  BOOST_LOG_TRIVIAL(debug) << "Pooling output size: " << volume_size(output_volume);

  return &output_volume;
};

void concatenator::concatenate(cl::sycl::queue q){
  for (int i=0; i<volumes_number;i++) {
    q.submit( [&] (handler &concatenategroup) { 
      BOOST_LOG_TRIVIAL(trace) << "CONCAT: submitting task " << i+1 << " to queue";
      auto output_a = concatenated_volume.get_access<access::mode::write>(concatenategroup);
      auto volumei_a = input_volumes[i]->get_access<access::mode::read>(concatenategroup);
      concatenategroup.parallel_for<class pad>( range<3>(output_width,output_height,input_depths[i]),
		[=] (id<3> index) {
		id<3> output_index=index+id<3>(0,0,offsets[i]);
		BOOST_LOG_TRIVIAL(trace) << "CONCAT: Writing element " << index_tostring(index) 
			<< " of volume " << i+1
			<< " (" 
			<< volumei_a[index]
			<< ") inside element " << index_tostring(output_index)
		        << " of output volume.";	
		output_a[output_index]=volumei_a[index];
      });
    });
  };
}
