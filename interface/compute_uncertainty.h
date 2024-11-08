#ifndef LMP_COMPUTE_UNCERTAINTY_H
#define LMP_COMPUTE_UNCERTAINTY_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeUncertainty : public Compute {
public:
  ComputeUncertainty(class LAMMPS *, int, char **);
  ~ComputeUncertainty();
  void init();
  double compute_scalar();

private:
  class PairCurator *pair;
};

}

#endif