#ifndef MISC_H
#define MISC_H

#include <CL/sycl.hpp>
#include <rang/rang.hpp>
#include <iostream>

#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>

namespace logging = boost::log;

typedef cl::sycl::buffer<float,3> Volume;
typedef std::vector<Volume> Weights;

void initialize_volume(Volume& v);
void initialize_volume(Volume& v, float val);
void initialize_volume(Volume& v, bool int_, int randmax);

void print_separator(rang::fg color, int length);
void print_volume(Volume&);
std::string volume_size(const Volume&);
std::string index_tostring(const cl::sycl::id<3> id);

std::vector<Volume> generate_stub_weights(size_t size,size_t depth,int filter_number);

Volume concatenate_volumes(std::vector<Volume>);

#endif
