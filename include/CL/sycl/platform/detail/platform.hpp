#ifndef TRISYCL_SYCL_PLATFORM_DETAIL_PLATFORM_HPP
#define TRISYCL_SYCL_PLATFORM_DETAIL_PLATFORM_HPP

/** \file The OpenCL SYCL abstract platform

    Ronan at Keryell point FR

    This file is distributed under the University of Illinois Open Source
    License. See LICENSE.TXT for details.
*/

#include "CL/sycl/detail/default_classes.hpp"

#include "CL/sycl/platform.hpp"

namespace cl {
namespace sycl {
namespace detail {

/** \addtogroup execution Platforms, contexts, devices and queues
    @{
*/

/// An abstract class representing various models of SYCL platforms
class platform {

public:

#ifdef TRISYCL_OPENCL
  /// Return the cl_platform_id of the underlying OpenCL platform
  virtual cl_platform_id get() const = 0;
#endif


  /// Return true if the platform is a SYCL host platform
  virtual bool is_host() const = 0;


  /// Query the platform for OpenCL string info::platform info
  virtual string_class get_info_string(info::platform param) const = 0;


  /// Specify whether a specific extension is supported on the platform.
  virtual bool has_extension(const string_class &extension) const = 0;


  // Virtual to call the real destructor
  virtual ~platform() {}

};

/// @} to end the execution Doxygen group

}
}
}

/*
    # Some Emacs stuff:
    ### Local Variables:
    ### ispell-local-dictionary: "american"
    ### eval: (flyspell-prog-mode)
    ### End:
*/

#endif // TRISYCL_SYCL_PLATFORM_DETAIL_PLATFORM_HPP
