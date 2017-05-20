#include <CL/sycl.hpp>
#include <convolution.hpp>
#include <iomanip>

const size_t WIDTH = 32;
const size_t HEIGHT = 32;
const size_t DEPTH = 3;

void print_volume(Volume v,size_t width, size_t height,size_t depth){
  auto V = v.get_access<access::mode::read>();
  for(int z=0; z<depth; z++){
    for(int y=0; y<height; y++){
      for(int x=0; x<width; x++){
	      std::cout << std::setfill('0') << std::setw(3) << V[x][y][z] << " ";
      }
      std::cout << std::endl;
    }
    std::cout << "_________________________________" << std::endl ;
  }
}

int main(){
  Volume input_volume = rand_input_generator(WIDTH,HEIGHT,DEPTH);
  print_volume(input_volume,WIDTH,HEIGHT,DEPTH);
  return 0;
};
