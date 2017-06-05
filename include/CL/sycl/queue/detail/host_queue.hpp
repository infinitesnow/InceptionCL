#ifndef TRISYCL_SYCL_QUEUE_DETAIL_HOST_QUEUE_HPP
#define TRISYCL_SYCL_QUEUE_DETAIL_HOST_QUEUE_HPP

/** \file Some implementation details of the host queue

    Ronan at Keryell point FR

    This file is distributed under the University of Illinois Open Source
    License. See LICENSE.TXT for details.
*/

#ifdef TRISYCL_OPENCL
#include <boost/compute.hpp>
#endif

#include "CL/sycl/context.hpp"
#include "CL/sycl/detail/debug.hpp"
#include "CL/sycl/device.hpp"
#include "CL/sycl/queue/detail/queue.hpp"

namespace cl {
namespace sycl {
namespace detail {

/** Some implementation details about the SYCL queue

    \todo Once a triSYCL queue is no longer blocking, make this a singleton
 */
class host_queue : public detail::queue,
                   detail::debug<host_queue> {

#ifdef TRISYCL_OPENCL
  /** Return the cl_command_queue of the underlying OpenCL queue

      This throws an error since there is no OpenCL queue associated
      to the host queue.
  */
  cl_command_queue get() const override {
    throw non_cl_error("The host queue has no OpenCL command queue");
  }


  /** Return the underlying Boost.Compute command queue

      This throws an error since there is no OpenCL queue associated
      to the host queue.
  */
  boost::compute::command_queue &get_boost_compute() override {
    throw non_cl_error("The host queue has no OpenCL command queue");
  }
#endif


  /// Return the SYCL host queue's host context
  cl::sycl::context get_context() const override {
    // Return the default context which is the host context
    return {};
  }


  /// Return the SYCL host device the host queue is associated with
  cl::sycl::device get_device() const override {
    // Return the default device which is the host device
    return {};
  }


  /// Claim proudly that the queue is executing on the SYCL host device
  bool is_host() const override {
    return true;
  }


};

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

#endif // TRISYCL_SYCL_QUEUE_DETAIL_HOST_QUEUE_HPP
