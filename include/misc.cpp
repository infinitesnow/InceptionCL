#include <misc.hpp>
#include <iomanip>
#include <ctime>

using namespace cl::sycl;

Volume rand_input_generator(size_t width, size_t height, size_t depth){
  
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

void print_volume(Volume &v,size_t width, size_t height,size_t depth){
  
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
    std::cout << stringbuffer[i].str();
    for (int j=0; j<4*width; j++) std::cout << "*";
    std::cout << std::endl;
  }
};

