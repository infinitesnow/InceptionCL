cmake_minimum_required ( VERSION 3.5 )
project ( InceptionModule CXX C )

# Include dependencies
find_package( Threads )
find_package( Boost REQUIRED COMPONENTS log)

add_executable( im "im.cpp" "include/convolution.cpp" "include/misc.cpp")
include_directories("${PROJECT_SOURCE_DIR}/include" "${Boost_INCLUDE_DIRS}")
target_link_libraries( im Threads::Threads "${Boost_LIBRARIES}") 

# Add compiler options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0") 
add_definitions(-DBOOST_LOG_DYN_LINK)
