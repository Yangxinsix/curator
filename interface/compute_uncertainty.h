#ifdef COMPUTE_CLASS
ComputeStyle(uncertainty,ComputeUncertainty);
#else

#ifndef LMP_COMPUTE_UNCERTAINTY_H
#define LMP_COMPUTE_UNCERTAINTY_H

#include "compute.h"

#include <string>

namespace LAMMPS_NS {

class Pair;

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
  double *uncertainty_ptr;
};

void clear_pair_uncertainties(Pair *);
void set_pair_uncertainty(Pair *, const std::string &, double);
double *get_pair_uncertainty_ptr(Pair *, const std::string &);

}

#endif
#endif
