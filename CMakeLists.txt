cmake_minimum_required ( VERSION 3.5 )
project ( InceptionModule )
add_executable( im im.cpp)

# Include dependencies
find_package( Threads )
find_package( Curses )
target_link_libraries( im Threads::Threads ncurses ) 
target_include_directories( im PRIVATE "${CMAKE_CURRENT_SOURCE_DIRECTORY}../triSYCL/include" )

# Add compiler options
add_compile_options("-g") # 
add_compile_options("-O0") # 

