#include <convolution.hpp>

//#define DEBUG_LEVEL warning 
#define DEBUG_LEVEL trace 

void init_boost()
{
    logging::core::get()->set_filter
    (
        logging::trivial::severity >= logging::trivial::DEBUG_LEVEL
    );
};

inline void print_header(std::string s){
  using namespace std;
  print_separator(rang::fg::red,s.length()+4);
  cout << rang::fg::red << "║ " << s << " ║" << rang::style::reset << endl; 
  print_separator(rang::fg::red,s.length()+4);
}

int main(){
  // Let's initialize Boost logging system
  init_boost();

  BOOST_LOG_TRIVIAL(trace) << "Starting main";

  // Define input parameters
  const size_t input_width = 14;
  const size_t input_height = 14;
  const size_t input_depth = 3;
  const short stride=1;
  const int bias = 0;
  
  /* 
   * GoogLeNet values
   *
  const int num11=128;
  const int num33reduce=128;
  const int num33=256;
  const int num55reduce=24;
  const int num55=64;
  const int poolproj=64; 
  */

  const int num1_11=5;
  const int num2_11=5;
  const int num2_33=3;
  const int num3_11=2;
  const int num3_55=6;
  const int num4_11=5; 


  /* 
   * Let's create the components of the Inception module. For each volume,
   * we first create a convolver object; weights are generated by a stub method.
   * Then, we apply the convolution/pool. 
  */
  cl::sycl::queue q;

  print_header("Initializing volumes");
  Volume input_volume=Volume(cl::sycl::range<3>(input_width,input_height,input_depth));
  Volume *vol1, *vol2, *vol3, *vol4, *vol2_t, *vol3_t, *vol4_t;

  initialize_volume(input_volume, true, 255, q);

  print_header("Initializing convolvers");
  Weights weights_1_11 = generate_stub_weights(1,input_depth,num1_11,q);
  convolver c1_11(weights_1_11,stride,bias);
  c1_11.initialize_hard(input_volume,q);

  Weights weights_2_11 = generate_stub_weights(1,input_depth,num2_11,q);
  convolver c2_11(weights_2_11,stride,bias);
  c2_11.initialize_hard(input_volume,q);
  Weights weights_2_33 = generate_stub_weights(3,num2_11,num2_33,q);
  convolver c2_33(weights_2_33,stride,bias);
  c2_33.initialize_soft(&vol2_t,input_width,input_height,num2_33,q);
  
  Weights weights_3_11 = generate_stub_weights(1,input_depth,num3_11,q);
  convolver c3_11(weights_3_11,stride,bias);
  c3_11.initialize_hard(input_volume,q);
  Weights weights_3_55 = generate_stub_weights(5,num3_11,num3_55,q);
  convolver c3_55(weights_3_55,stride,bias);
  c3_55.initialize_soft(&vol3_t,input_width,input_height,num3_55,q);

  convolver p4_33(3,1);
  p4_33.initialize_hard(input_volume,q);
  Weights weights_4_11 = generate_stub_weights(1,input_depth,num4_11,q);
  convolver c4_11(weights_4_11,stride,bias);
  c4_11.initialize_soft(&vol4_t,input_width,input_height,num4_11,q);

  //std::this_thread::sleep_for(std::chrono::seconds(3));

  print_header("Launching convolutions");
  c1_11.convolve();

  vol2_t=c2_11.convolve();
  vol2=c2_33.convolve();

  vol3_t=c3_11.convolve();
  vol3=c3_55.convolve();

  vol4_t=p4_33.pool();
  vol4=c4_11.convolve();
  
  //print_header("Concatenating");
  //Volume output = concatenate_volumes(Weights{*vol1,*vol2,*vol3,*vol4},q);
  //print_header("Final output");
  //print_volume(output);
 
  std::this_thread::sleep_for(std::chrono::seconds(3));
  
  std::cout << "Finished." << std::endl;
  return 0;
};
