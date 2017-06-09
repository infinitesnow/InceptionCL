#include <misc.hpp>
#include <iomanip>
#include <ctime>

using namespace cl::sycl;

void print_separator(rang::fg color, int length){
  std::cout << color;
  for (int j=0; j<length; j++)  std::cout << "═";
  std::cout << rang::style::reset << std::endl;  
}

void print_volume(Volume &v){
  using namespace std;
  BOOST_LOG_TRIVIAL(trace) << "MISCPRINT: Printing volume " << volume_size(v);
  const short item_length = 8;

  size_t width = v.get_range().get(0);
  size_t height = v.get_range().get(1);
  size_t depth = v.get_range().get(2);
  
  BOOST_LOG_TRIVIAL(trace) << "MISCPRINT: Requesting access to buffer";
  auto V = v.get_access<access::mode::read>();
  BOOST_LOG_TRIVIAL(trace) << "MISCPRINT: Access granted";
  
  cout.setf(ios::fixed);
  cout.precision(2);

  int separator_length = (item_length+1)*width+2;
  print_separator(rang::fg::blue,separator_length);
  print_separator(rang::fg::cyan,separator_length);
  for(int z=0; z<depth; z++){
    for(int y=0; y<height; y++){
      cout << rang::fg::cyan << "║" << rang::style::reset;
      for(int x=0; x<width; x++){
	      cout << setfill(' ') << rang::fg::gray << setw(item_length) << V[x][y][z] << " ";
      }
      cout << rang::fg::cyan << "║" << rang::style::reset <<endl;
    }
    print_separator(rang::fg::cyan,separator_length);
  }
  print_separator(rang::fg::blue,separator_length);
  cout << endl;
};

std::string volume_size(Volume const& v){
  std::stringstream out; 
  out << v.get_range().get(0) << "x" << v.get_range().get(1) << "x" << v.get_range().get(2);
  return out.str();
};

std::string index_tostring(cl::sycl::id<3> const id){
  std::stringstream out;
  out << "(" << id[0] << "," << id[1] << "," << id[2] <<")";
  return out.str();
}

inline void initialize_volume_inline(Volume &v, float val, bool random, bool int_, int randmax, cl::sycl::queue q) {
   BOOST_LOG_TRIVIAL(trace) << "MISCINIT: Initializing volume " << volume_size(v) 
	   << ", random: " << random
	   << ", integer: " << int_ 
	   << ", randmax: " << randmax;
   size_t width = v.get_range().get(0);
   size_t height = v.get_range().get(1);
   size_t depth = v.get_range().get(2);

   q.submit( [&] (handler &cmdgroup) { 
     BOOST_LOG_TRIVIAL(trace) << "MISCINIT: Submitting to queue for volume of size " << volume_size(v);
     auto v_a = v.get_access<access::mode::write>(cmdgroup);
     cmdgroup.parallel_for<class pad>( range<3>(width,height,depth),
         	    [=] (id<3> index) {
		float tmp = !random ? float(val) : ( int_ ? int(rand()%randmax) : ((float(rand())/RAND_MAX)*randmax));
		BOOST_LOG_TRIVIAL(trace) << "MISCINIT: Initializing " << index_tostring(index)
			<< " element of volume of size " << volume_size(v)
			<< " with " << tmp;
		v_a[index] = tmp;
     });
   });
};

void initialize_volume(Volume& v, cl::sycl::queue q){
	initialize_volume_inline(v, 0, true, false, 1, q);
}
void initialize_volume(Volume& v, float val, cl::sycl::queue q){
	initialize_volume_inline(v, 0, false, false, 1, q);
}
void initialize_volume(Volume& v, bool int_, int randmax, cl::sycl::queue q){
	initialize_volume_inline(v, 0, true, int_, randmax, q);
}	

std::vector<Volume> generate_stub_weights(size_t size,size_t depth,int filter_number, cl::sycl::queue q) {
  BOOST_LOG_TRIVIAL(debug) << "MISC: Generating stub weights for " << filter_number << 
	  " filters (" << size << "x" << size << "x" << depth << ")...";

  std::vector<Volume> weights_vector;
  for (int i=0; i<filter_number; i++){
    BOOST_LOG_TRIVIAL(debug) << "MISC: Generating weights for filter " << i;
    Volume w( cl::sycl::range<3>(size,size,depth));      
    initialize_volume(w,q);
    //std::cout << "Stub weights for filter " << i+1 << ":" <<std::endl;
    //print_volume(w);
    weights_vector.push_back(w);
  }
  
  return weights_vector;
};

Volume concatenate_volumes(std::vector<Volume> input_volumes, cl::sycl::queue q){
  BOOST_LOG_TRIVIAL(info) << "CONCAT: Concatenating volumes...";
  int volumes_number=input_volumes.size();
  BOOST_LOG_TRIVIAL(info) << "CONCAT: Concatenating " << volumes_number << " volumes";
  
  size_t output_width=input_volumes[0].get_range().get(0);
  size_t output_height=input_volumes[0].get_range().get(1);
  std::vector<size_t> input_depths;
  for (int i=0; i<volumes_number; i++) {
    BOOST_LOG_TRIVIAL(trace) << "CONCAT: Volume " << i+1 <<" is of size (" 
	    << input_volumes[i].get_range().get(0) << ","
	    << input_volumes[i].get_range().get(1) << ","
	    << input_volumes[i].get_range().get(2) << ")";
    input_depths.push_back(input_volumes[i].get_range().get(2));
  }
  size_t output_depth=std::accumulate(input_depths.begin(),input_depths.end(),0);

  BOOST_LOG_TRIVIAL(trace) << "CONCAT: Concatenation output is of size ("
	  << output_width << ","
	  << output_height << ","
	  << output_depth <<")";

  
  std::vector<size_t> offsets(volumes_number);
  for (int i=0; i<volumes_number; i++) {
    int offset=std::accumulate(input_depths.begin(), input_depths.begin()+i, 0);
    BOOST_LOG_TRIVIAL(trace) << "CONCAT: Computed " << i+1 << "^ offset: " << offset;
    offsets[i]=offset; 
  } 


  Volume concatenated_volume(range<3>(output_width,output_height,output_depth));	
  
  for (int i=0; i<volumes_number; i++) {
    q.submit( [&] (handler &concatenategroup) { 
	BOOST_LOG_TRIVIAL(trace) << "CONCAT: submitting task " << i+1 << " to queue";
        auto output_a = concatenated_volume.get_access<access::mode::write>(concatenategroup);
        auto volumei_a = input_volumes[i].get_access<access::mode::read>(concatenategroup);
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
  }

  return concatenated_volume;
}
