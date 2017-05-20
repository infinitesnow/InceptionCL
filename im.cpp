#include <CL/sycl.hpp>
#include <convolution.hpp>

const size_t WIDTH = 32;
const size_t HEIGHT = 32;
const size_t DEPTH = 3;

void print_volume(Volume v,width,height,depth){
  for(int z=0; z<depth; z++){
    for(int y=0; y<height; y++){
      for(int x=0; x<width; x++){
	      std::cout << v[x][y][z] << " ";
      }
    }
    std::cout << std::endl;
  }
}

int main(){
	Volume input_volume = rand_input_generator(WIDTH,HEIGHT,DEPTH);
	print_volume(input_volume);
	return 0;
};
