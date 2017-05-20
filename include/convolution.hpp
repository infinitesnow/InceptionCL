#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include <CL/sycl.hpp>
template<typename T>
struct volume {
  vector<T> R;
  vector<T> G;
  vector<T> B;	
}

class conv_functor(int, int, int, volume){
  int conv_size;
  int stride;
  int padding;
  volume operator() (volume);
  private:
    volume weights;  
};

class maxpool_functor(int, volume){
  int pool_size;
  int stride;
  int padding;
  volume operator() (volume);
};

#endif
