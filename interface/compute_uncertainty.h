#ifndef LMP_COMPUTE_UNCERTAINTY_H
#define LMP_COMPUTE_UNCERTAINTY_H

ComputeStyle(uncertainty, ComputeUncertainty)

#include "compute.h"

namespace LAMMPS_NS {

class ComputeUncertainty : public Compute {
public:
  ComputeUncertainty(class LAMMPS *, int, char **);
  ~ComputeUncertainty();
  void init();
  double compute_scalar();

 private:
  std::string uncertainty_name; // Name of the uncertainty to extract
  double *uncertainty_value;    // Pointer to the uncertainty value
  class Pair *pair_ptr; 
};

}

#endif