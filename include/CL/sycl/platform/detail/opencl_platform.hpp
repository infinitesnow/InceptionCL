#ifndef TRISYCL_SYCL_PLATFORM_DETAIL_OPENCL_PLATFORM_HPP
#define TRISYCL_SYCL_PLATFORM_DETAIL_OPENCL_PLATFORM_HPP

/** \file The OpenCL triSYCL OpenCL platform implementation

    Ronan at Keryell point FR

    This file is distributed under the University of Illinois Open Source
    License. See LICENSE.TXT for details.
*/
#include <memory>

#include <boost/compute.hpp>

#include "CL/sycl/detail/default_classes.hpp"

#include "CL/sycl/detail/cache.hpp"
#include "CL/sycl/detail/unimplemented.hpp"
#include "CL/sycl/device.hpp"
#include "CL/sycl/exception.hpp"
#include "CL/sycl/info/param_traits.hpp"
#include "CL/sycl/platform/detail/platform.hpp"

namespace cl {
namespace sycl {

class device;

namespace detail {

/** \addtogroup execution Platforms, contexts, devices and queues
    @{
*/

/// SYCL OpenCL platform
class opencl_platform : public detail::platform {

  /// Use the Boost Compute abstraction of the OpenCL platform
  boost::compute::platform p;

  /** A cache to always return the same live platform for a given OpenCL
      platform

      C++11 guaranties the static construction is thread-safe
  */
  static detail::cache<cl_platform_id, detail::opencl_platform> cache;

public:

  /// Return the cl_platform_id of the underlying OpenCL platform
  cl_platform_id get() const override {
    return p.id();
  }


  /// Return false since an OpenCL platform is not the SYCL host platform
  bool is_host() const override {
    return false;
  }


#if 0
  /** Returns at most the host device for this platform, according to
      the requested kind

      By default returns all the devices, which is obviously the host
      one here

      \todo To be implemented
  */
  vector_class<cl::sycl::device>
  get_devices(info::device_type device_type = info::device_type::all)
    const override
  {
    detail::unimplemented();
    return {};
  }
#endif


  /// Returning the information string parameters for the OpenCL platform
  string_class get_info_string(info::platform param) const override {
    /* Use the fact that the triSYCL info values are the same as the
       OpenCL ones used in Boost.Compute to just cast the enum class
       to the int value */
    return p.get_info<std::string>(static_cast<cl_platform_info>(param));
  }


  /// Specify whether a specific extension is supported on the platform
  bool has_extension(const string_class &extension) const override {
    return p.supports_extension(extension);
  }


  ///// Get a singleton instance of the opencl_platform
  static std::shared_ptr<opencl_platform>
  instance(const boost::compute::platform &p) {
    return cache.get_or_register(p.id(),
                                 [&] { return new opencl_platform { p }; });
  }

private:

  /// Only the instance factory can built it
  opencl_platform(const boost::compute::platform &p) : p { p } {}

public:

  /// Unregister from the cache on destruction
  ~opencl_platform() override {
    cache.remove(p.id());
  }

};

/* Allocate the cache here but since this is a pure-header library,
   use a weak symbol so that only one remains when SYCL headers are
   used in different compilation units of a program
*/
TRISYCL_WEAK_ATTRIB_PREFIX
detail::cache<cl_platform_id, detail::opencl_platform> opencl_platform::cache
TRISYCL_WEAK_ATTRIB_SUFFIX;

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

#endif // TRISYCL_SYCL_PLATFORM_DETAIL_HOST_PLATFORM_HPP
