#include <CL/sycl.hpp>
#include <iostream>

using namespace cl::sycl;

#define TOL (0.001)   // tolerance used in floating point comparisons
#define LENGTH (1024) // Length of vectors a, b and c

int main() {
  
  const std::size_t array_size = LENGTH;

  std::vector<float> h_a(array_size);             // a vector
  std::vector<float> h_b(array_size);             // b vector
  std::vector<float> h_c(array_size);             // c vector
  std::vector<float> h_r(array_size); 	      // d vector (result)

  // Fill vectors a and b with random float values
  for (int i = 0; i < array_size; i++) {
    h_a[i] = rand() / (float)RAND_MAX;
    h_b[i] = rand() / (float)RAND_MAX;
    h_c[i] = rand() / (float)RAND_MAX;
    h_r[i] = 0;
  }

  {
    // Create a queue to work on 
    queue myQueue;

    // Create range
    range<1> numOfItems(array_size);

    // Device buffers
    buffer<float> d_a(h_a.data(), numOfItems);
    buffer<float> d_b(h_b.data(), numOfItems);
    buffer<float> d_c(h_c.data(), numOfItems);
    buffer<float> d_r(h_r.data(), numOfItems);
    
    myQueue.submit( [&](handler &cmdgroup)
    {

     // Data accessors
     auto a = d_a.get_access<access::mode::read>(cmdgroup);
     auto b = d_b.get_access<access::mode::read>(cmdgroup);
     auto c = d_c.get_access<access::mode::read>(cmdgroup);
     auto r = d_r.get_access<access::mode::write>(cmdgroup); 

     // Kernel
     cmdgroup.parallel_for<class sum_kernel>( numOfItems,
       [=](id<1> item) {
       r[item] = a[item] + b[item] + c[item];
       }
     );
    });
  }


  // Test the results
  int correct = 0;
  float tmp;
  //auto res=d_r.get_access<access::mode::read>();
  for (int i = 0; i < array_size; i++) {
    tmp = h_a[i] + h_b[i] + h_c[i]; 	// assign element i of a+b+c to tmp
    tmp -= h_r[i]; 			// compute deviation of expected and output result
    if (tmp * tmp < TOL * TOL) 		// correct if square deviation is less than tolerance squared
      correct++;
    else
      printf("[%d] tmp %f h_a %f h_b %f h_c %f h_r %f \n", i, tmp, h_a[i], h_b[i], h_c[i], h_r[i]);
  }
  // summarize results
  printf("R = A+B+C: %d out of %d results were correct.\n", correct, array_size);
  return (correct == array_size);

}
