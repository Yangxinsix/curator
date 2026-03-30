#ifdef COMPUTE_CLASS
ComputeStyle(uncertainty/atom,ComputeUncertaintyAtom);
#else

#ifndef LMP_COMPUTE_UNCERTAINTY_ATOM_H
#define LMP_COMPUTE_UNCERTAINTY_ATOM_H

#include "compute.h"

#include <string>
#include <vector>

namespace LAMMPS_NS {

class Pair;

class ComputeUncertaintyAtom : public Compute {
public:
  ComputeUncertaintyAtom(class LAMMPS *, int, char **);
  ~ComputeUncertaintyAtom();
  void init();
  void compute_peratom();
  double memory_usage();

private:
  std::string uncertainty_name;
  int debug_mode;
  int nmax;
  class Pair *pair_ptr;
  double *uncertainty_array;
};

void clear_pair_uncertainty_arrays(Pair *);
void set_pair_uncertainty_array(Pair *, const std::string &, const double *, int);
double *get_pair_uncertainty_array_ptr(Pair *, const std::string &, int &);

}

#endif
#endif
