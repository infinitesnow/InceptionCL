/** \file

    \mainpage

    This is the main OpenCL SYCL C++ header file to experiment with
    the OpenCL CL provisional specification.

    For more information about OpenCL SYCL:
    http://www.khronos.org/sycl/

    For more information on this project and to access to the source of
    this file, look at https://github.com/triSYCL/triSYCL

    The Doxygen version of the implementation itself is in
    http://Xilinx.github.io/triSYCL/Doxygen/triSYCL/html and
    http://Xilinx.github.io/triSYCL/Doxygen/triSYCL/triSYCL-implementation-refman.pdf


    Ronan at keryell dot FR

    Copyright 2014--2015 Advanced Micro Devices, Inc.

    Copyright 2015--2017 Xilinx, Inc.

    This file is distributed under the University of Illinois Open Source
    License. See LICENSE.TXT for details.
*/


/** Some global triSYCL configuration */
#include "CL/sycl/detail/global_config.hpp"
#include "CL/sycl/detail/default_classes.hpp"


/* All the SYCL components, one per file */
#include "CL/sycl/access.hpp"
#include "CL/sycl/accessor.hpp"
#include "CL/sycl/allocator.hpp"
#include "CL/sycl/address_space.hpp"
#include "CL/sycl/buffer.hpp"
#include "CL/sycl/context.hpp"
#include "CL/sycl/device.hpp"
#include "CL/sycl/device_selector.hpp"
#include "CL/sycl/error_handler.hpp"
#include "CL/sycl/event.hpp"
#include "CL/sycl/exception.hpp"
#include "CL/sycl/group.hpp"
#include "CL/sycl/handler.hpp"
#include "CL/sycl/id.hpp"
#include "CL/sycl/image.hpp"
#include "CL/sycl/item.hpp"
#include "CL/sycl/math.hpp"
#include "CL/sycl/nd_item.hpp"
#include "CL/sycl/nd_range.hpp"
#include "CL/sycl/opencl_type.hpp"
#include "CL/sycl/parallelism.hpp"
#include "CL/sycl/pipe.hpp"
#include "CL/sycl/pipe_reservation.hpp"
#include "CL/sycl/platform.hpp"
#include "CL/sycl/queue.hpp"
#include "CL/sycl/range.hpp"
#include "CL/sycl/static_pipe.hpp"
#include "CL/sycl/vec.hpp"

// Some includes at the end to break some dependencies
#include "CL/sycl/device_selector/detail/device_selector_tail.hpp"
#include "CL/sycl/device/detail/device_tail.hpp"

/*
    # Some Emacs stuff:
    ### Local Variables:
    ### ispell-local-dictionary: "american"
    ### eval: (flyspell-prog-mode)
    ### End:
*/
