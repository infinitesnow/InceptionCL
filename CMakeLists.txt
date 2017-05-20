cmake_minimum_required ( VERSION 3.5 )
project ( InceptionModule CXX C )
include_directories("${PROJECT_SOURCE_DIR}/include")

# Include dependencies
find_package( Threads )
find_package( Curses )

add_executable( im im.cpp include/convolution.cpp)
target_link_libraries( im Threads::Threads ncurses ) 

add_executable( cursestest cursestest.cpp)
target_link_libraries( cursestest ncurses ) 

# Add compiler options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0") 
