#include <misc.hpp>
#include <iomanip>
#include <ctime>

using namespace cl::sycl;

Volume rand_volume_generator(size_t width, size_t height, size_t depth){
  
  queue q;

  std::cout << "Generating random input volume..." << std::endl;

  Volume v( range<3>(width,height,depth) );

  {
    q.submit( [&] (handler &cmdgroup) {
      auto V = v.get_access<access::mode::write>(cmdgroup);
      cmdgroup.parallel_for<class rand_init>( range<3>(width,height,depth),
          	    [=] (id<3> index) {
          	    V[index]=rand()%256;
      });
    });
  }

  return v;
};

void print_volume(Volume &v){

  const short item_length = 8;

  size_t width = v.get_range().get(0);
  size_t height = v.get_range().get(1);
  size_t depth = v.get_range().get(2);
  
  auto V = v.get_access<access::mode::read>();
  
  std::cout.setf(std::ios::fixed);
  std::cout.precision(2);

  for(int z=0; z<depth; z++){
    std::cout << "Layer " << z+1 << ":" << std::endl;
    for(int y=0; y<height; y++){
      for(int x=0; x<width; x++){
	      std::cout << std::setfill(' ') << std::setw(item_length) << V[x][y][z] << " ";
      }
      std::cout << std::endl;
    }
    for (int j=0; j<(item_length+1)*width-1; j++) std::cout << "*";
    std::cout << std::endl;
  }
};

std::string volume_size(Volume& v){
  std::stringstream out; 
  out << v.get_range().get(0) << "x" << v.get_range().get(1) << "x" << v.get_range().get(2);
  return out.str();
};

void initialize_volume(Volume &v, float val) {
   size_t width = v.get_range().get(0);
   size_t height = v.get_range().get(1);
   size_t depth = v.get_range().get(2);

   queue q;

   q.submit( [&] (handler &cmdgroup) { 
     auto v_a = v.get_access<access::mode::write>(cmdgroup);
     cmdgroup.parallel_for<class pad>( range<3>(width,height,depth),
         	    [=] (id<3> index) {
       	    v_a[index]=val;
     });
   });
};

void initialize_volume(Volume& v) {
   size_t width = v.get_range().get(0);
   size_t height = v.get_range().get(1);
   size_t depth = v.get_range().get(2);

   queue q;

   q.submit( [&] (handler &cmdgroup) { 
     auto v_a = v.get_access<access::mode::write>(cmdgroup);
     cmdgroup.parallel_for<class pad>( range<3>(width,height,depth),
         	    [=] (id<3> index) {
       	    v_a[index]=( (float) rand()/RAND_MAX);
     });
   });
};

std::vector<Volume> generate_stub_weights(size_t size,size_t depth,int filter_number) {
  std::cout << "Generating stub weights for " << filter_number << 
	  " filters (" << size << "x" << size << "x" << depth << ")..." << std::endl;

  std::vector<Volume> weights_vector;
  for (int i=0; i<filter_number; i++){
    Volume w( cl::sycl::range<3>(size,size,depth));      
    initialize_volume(w);
    //std::cout << "Stub weights for filter " << i+1 << ":" <<std::endl;
    //print_volume(w);
    weights_vector.push_back(w);
  }
  
  return weights_vector;
};

Volume concatenate_volumes(std::vector<Volume> input_volumes){
  int volumes_number=input_volumes.size();
  size_t output_width=input_volumes[0].get_range().get(0);
  size_t output_height=input_volumes[0].get_range().get(1);
  std::vector<size_t> input_depths;
  for (int i=0; i<volumes_number; i++) {
    input_depths.push_back(input_volumes[i].get_range().get(2));
  }
  size_t output_depth=std::accumulate(input_depths.begin(),input_depths.end(),0);
  std::vector<size_t> offsets(volumes_number);
  for (int i=0; i<volumes_number; i++) {
    std::partial_sum(&input_depths[0], &input_depths[i], &offsets[i]);
  } 

  Volume concatenated_volume(range<3>(output_width,output_height,output_depth));	
  
  queue q;
  for (int i=0; i<volumes_number; i++) {
    q.submit( [&] (handler &concatenategroup) { 
        auto output_a = concatenated_volume.get_access<access::mode::write>(concatenategroup);
        auto volumei_a = input_volumes[i].get_access<access::mode::read>(concatenategroup);
        concatenategroup.parallel_for<class pad>( range<3>(output_width,output_height,input_depths[i]),
			[=] (id<3> index) {
			id<3> output_index=index+id<3>(0,0,offsets[i]);
			output_a[output_index]=volumei_a[index];
      });
    });
  }

  return concatenated_volume;
}
