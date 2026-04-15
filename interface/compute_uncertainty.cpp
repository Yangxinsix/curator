#include "compute_uncertainty.h"
#include "atom.h"
#include "error.h"
#include "force.h"
#include "modify.h"
#include "pair_mliap.h"
#include "update.h"
#include <cstring>
#include <cstdlib>
#include <iostream>

using namespace LAMMPS_NS;

ComputeUncertainty::ComputeUncertainty(LAMMPS *lmp, int narg, char **arg)
  : Compute(lmp, narg, arg) {
  if (narg != 4)
    error->all(FLERR, "Illegal compute uncertainty command");

  uncertainty_name = std::string(arg[3]);

  scalar_flag = 1;
  extscalar = 0;
  
  debug_mode = 0;

  // initialize pair and uncertainty pointer
  pair_ptr = NULL;
  uncertainty_ptr = nullptr;
  if(const char* env_p = std::getenv("CURATOR_DEBUG")){
    debug_mode = 1;
  }
}

void ComputeUncertainty::init() {
  pair_ptr = force->pair;
  if (pair_ptr == NULL)
    error->all(FLERR, "Compute uncertainty requires a pair style to be defined");

  int extract_dim = 0;
  uncertainty_ptr = static_cast<double *>(pair_ptr->extract(uncertainty_name.c_str(), extract_dim));
  if (uncertainty_ptr != NULL) return;

  int peratom_cols = 0;
  if (pair_ptr->extract_peratom(uncertainty_name.c_str(), peratom_cols) != nullptr) {
    error->all(
      FLERR,
      "Compute uncertainty requested a per-atom uncertainty; use compute uncertainty/atom instead"
    );
  }

  PairMLIAP *pair_mliap = dynamic_cast<PairMLIAP *>(pair_ptr);
  if (pair_mliap != NULL) {
    uncertainty_ptr = pair_mliap->ensure_uncertainty_ptr(uncertainty_name);
    return;
  }

  error->all(
    FLERR,
    "Compute uncertainty requires pair style 'curator', 'mliap', or a pair style "
    "that exposes the requested key through Pair::extract()"
  );
}

double ComputeUncertainty::compute_scalar() {
  invoked_scalar = update->ntimestep;
  scalar = *uncertainty_ptr;
  if (debug_mode) {
    std::cout << "Key: " << uncertainty_name << ", Value: " << scalar << std::endl;
    std::cout << "Invoked Scalar: " << invoked_scalar << std::endl;
  }
  return scalar;
}

double ComputeUncertainty::memory_usage() {
  double bytes = sizeof(double);
  return bytes;
}
