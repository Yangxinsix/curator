#include "compute_uncertainty.h"
#include "atom.h"
#include "error.h"
#include "force.h"
#include "modify.h"
#include "pair_curator.h"
#include "pair_hybrid.h"
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

  // Check if the pair style is PairCurator
  pair_curator = dynamic_cast<PairCurator *>(pair_ptr);
  // check hybrid pair style
  if (pair_curator == NULL) {
    // Check if pair style is hybrid
    if (strcmp(pair_ptr->style, "hybrid") == 0) {
      // Search for PairCURATOR within hybrid
      PairHybrid *hybrid = dynamic_cast<PairHybrid *>(pair_ptr);
      int nstyles = hybrid->nstyles;
      for (int i = 0; i < nstyles; ++i) {
        pair_curator = dynamic_cast<PairCURATOR *>(hybrid->styles[i]);
        if (pair_curator != NULL) break;
      }
    }
  }

  if (pair_curator == NULL)
    error->all(FLERR, "Compute uncertainty can only be used with pair style 'curator'");
}

double ComputeUncertainty::compute_scalar() {
  invoked_scalar = update->ntimestep;
  double value = pair_curator->get_uncertainty(uncertainty_name);
  return value;
}

double ComputeUncertainty::memory_usage() {
  double bytes = sizeof(double);
  return bytes;
}