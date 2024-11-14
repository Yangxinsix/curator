#ifdef COMPUTE_CLASS
ComputeStyle(uncertainty,ComputeUncertainty);
#else

#ifndef LMP_COMPUTE_UNCERTAINTY_H
#define LMP_COMPUTE_UNCERTAINTY_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeUncertainty : public Compute {
public:
  ComputeUncertainty(class LAMMPS *, int, char **);
  ~ComputeUncertainty() {};
  void init();
  double compute_scalar();
  double memory_usage();

 private:
  std::string uncertainty_name; // Name of the uncertainty to extract
  int debug_mode;
  class Pair *pair_ptr; 
  class PairCurator *pair_curator;
};

}

#endif
#endif