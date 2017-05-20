cmake_minimum_required ( VERSION 3.5 )
project ( InceptionModule )
set(CMAKE_BINARY_DIR "${CMAKE_SOURCE_DIRECTORY}/build")
include_directories("${CMAKE_SOURCE_DIRECTORY}/include")

# Include dependencies
find_package( Threads )
find_package( Curses )

add_executable( im im.cpp)
target_link_libraries( im Threads::Threads ncurses ) 
target_include_directories( im PRIVATE "${CMAKE_SOURCE_DIRECTORY}../triSYCL/include" )

add_executable( cursestest cursestest.cpp)
target_link_libraries( cursestest ncurses ) 

# Add compiler options
add_compile_options("-g") 
add_compile_options("-O0")

