#include <misc.hpp>
#include <iomanip>
#include <ctime>

using namespace cl::sycl;

Volume rand_volume_generator(size_t width, size_t height, size_t depth){
  
  queue q;

  std::cout << "Generating random input volume" << std::endl;

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

void initialize_volume(Volume &v) {
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
  std::vector<Volume> weights_vector;
  for (int i=0; i<filter_number; i++){
    std::cout << "Generating stub weights for filter " << i+1 << std::endl;
    Volume w( cl::sycl::range<3>(size,size,depth));      
    initialize_volume(w);
    print_volume(w);
    weights_vector.push_back(w);
  }
  return weights_vector;
};
