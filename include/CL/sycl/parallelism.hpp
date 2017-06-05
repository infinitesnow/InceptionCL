#ifndef TRISYCL_SYCL_PARALLELISM_HPP
#define TRISYCL_SYCL_PARALLELISM_HPP

/** \file

    Implement parallel constructions to launch kernels

    Ronan at keryell dot FR

    This file is distributed under the University of Illinois Open Source
    License. See LICENSE.TXT for details.
*/

#include "CL/sycl/parallelism/detail/parallelism.hpp"
#include "CL/sycl/id.hpp"

namespace cl {
namespace sycl {

/** \addtogroup parallelism
    @{
*/

/// SYCL parallel_for version that allows a Program object to be specified
/// \todo To be implemented
/* template <typename Range, typename Program, typename ParallelForFunctor>
void parallel_for(Range r, Program p, ParallelForFunctor f) {
  /// \todo deal with Program
  parallel_for(r, f);
}
*/

  /** Loop on the work-items inside a work-group

      \todo Deprecate this function in the specification to use
      instead the group method
  */
  template <int Dimensions = 1, typename ParallelForFunctor>
  void parallel_for_work_item(const group<Dimensions> &g,
                              ParallelForFunctor f) {
    g.parallel_for_work_item(f);
  }



}
}

/// @} End the parallelism Doxygen group

/*
    # Some Emacs stuff:
    ### Local Variables:
    ### ispell-local-dictionary: "american"
    ### eval: (flyspell-prog-mode)
    ### End:
*/

#endif // TRISYCL_SYCL_PARALLELISM_HPP
