#include <misc.hpp>
#include <iomanip>
#include <ctime>

using namespace cl::sycl;

Volume rand_volume_generator(size_t width, size_t height, size_t depth){
  
  queue q;

  std::cout << "Generating random input volume" << std::endl;

  clock_t time_a = clock();

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

  clock_t time_b = clock();

  std::cout << "Operation completed in " << time_b-time_a << " ticks." << std::endl;

  return v;
};

void print_volume(Volume &v){

  size_t width = v.get_range().get(0);
  size_t height = v.get_range().get(1);
  size_t depth = v.get_range().get(2);
  
  auto V = v.get_access<access::mode::read>();
  
  std::stringstream stringbuffer[depth];

  for(int z=0; z<depth; z++){
    for(int y=0; y<height; y++){
      for(int x=0; x<width; x++){
	      stringbuffer[z] << std::setfill('0') << std::setw(3) << V[x][y][z] << " ";
      }
      stringbuffer[z] << std::endl;
    }
  }

  for ( int i=0; i<depth; i++){
    std::cout << "Layer " << i+1 << ":" << std::endl;
    std::cout << stringbuffer[i].str();
    for (int j=0; j<4*width; j++) std::cout << "*";
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

Volume generate_stub_weights(size_t size,size_t depth, float val) {
  std::cout << "Generating stub weights" << std::endl;
  //Volume w = rand_volume_generator(size,size,depth);
  Volume w = cl::sycl::buffer<float,3>( cl::sycl::range<3>(size,size,depth));      
  initialize_volume(w,val);
  print_volume(w);
  return w;
};
