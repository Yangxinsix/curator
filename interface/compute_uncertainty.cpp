#include "compute_uncertainty.h"
#include "error.h"
#include "force.h"
#include "pair_curator.h"

using namespace LAMMPS_NS;

ComputeUncertainty::ComputeUncertainty(LAMMPS *lmp, int narg, char **arg)
  : Compute(lmp, narg, arg) {
  if (narg != 3)
    error->all(FLERR, "Illegal compute uncertainty command");

  // Ensure the pair style is PairMyCustom
  pair = dynamic_cast<PairCurator *>(force->pair_match("mycustom", 0));
  if (!pair)
    error->all(FLERR, "Pair style mycustom is not active");
}

void ComputeUncertainty::init() {
  // Ensure pair style is properly initialized
  if (!pair)
    error->all(FLERR, "Pair style curator is not active");
}

double ComputeUncertainty::compute_scalar() {
  invoked_scalar = update->ntimestep;

  // Access the uncertainty data from the pair style
  return pair->uncertainty_scalar;
}