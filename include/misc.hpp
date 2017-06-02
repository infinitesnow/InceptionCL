#ifndef MISC_H
#define MISC_H

#include <CL/sycl.hpp>
#include <iostream>
#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
namespace logging = boost::log;

typedef cl::sycl::buffer<float,3> Volume;

std::vector<Volume> generate_stub_weights(size_t size,size_t depth,int filter_number);
void initialize_volume(Volume &v, float val);
void initialize_volume(Volume &v);
Volume rand_volume_generator(size_t width, size_t height, size_t depth);
void print_volume(Volume&);
std::string volume_size(Volume&);
Volume concatenate_volumes(std::vector<Volume>);
#endif
