#ifndef MISC_H
#define MISC_H

#include <CL/sycl.hpp>

typedef cl::sycl::buffer<float,3> Volume;

Volume rand_input_generator(size_t width, size_t height, size_t depth);
void print_volume(Volume &v,size_t width, size_t height,size_t depth);
#endif
