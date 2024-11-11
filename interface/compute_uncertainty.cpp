#include "compute_uncertainty.h"
#include "atom.h"
#include "error.h"
#include "force.h"
#include "modify.h"
#include "pair_curator.h"
#include "update.h"
#include <cstring>

using namespace LAMMPS_NS;

ComputeUncertainty::ComputeUncertainty(LAMMPS *lmp, int narg, char **arg)
  : Compute(lmp, narg, arg) {
  if (narg != 4)
    error->all(FLERR, "Illegal compute uncertainty command");

  uncertainty_name = std::string(arg[3]);

  scalar_flag = 1;
  extscalar = 0;

  // initialize pair and uncertainty pointer
  uncertainty_value = NULL;
  pair_ptr = NULL;
}

void ComputeUncertainty::init() {
  // Ensure pair style is properly initialized
  pair_ptr = force->pair;
  if (pair_ptr == NULL)
    error->all(FLERR, "Compute uncertainty requires a pair style to be defined");

  int dim;
  uncertainty_value = (double *)pair_ptr->extract(uncertainty_name.c_str(), dim);
  if (uncertainty_value == NULL || dim != 0)
    error->all(FLERR, "Uncertainty not found in pair style or invalid uncertainty name");
}

double ComputeUncertainty::compute_scalar() {
  invoked_scalar = update->ntimestep;
  return *uncertainty_value;
}