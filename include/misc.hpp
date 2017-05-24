#ifndef MISC_H
#define MISC_H

#include <CL/sycl.hpp>
#include <iostream>

typedef cl::sycl::buffer<float,3> Volume;

Volume generate_stub_weights(size_t size,size_t depth);
void initialize_volume(Volume &v, float val);
void initialize_volume(Volume &v);
Volume rand_volume_generator(size_t width, size_t height, size_t depth);
void print_volume(Volume &v);
#endif
