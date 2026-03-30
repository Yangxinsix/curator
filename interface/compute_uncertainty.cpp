#include "compute_uncertainty.h"
#include "atom.h"
#include "error.h"
#include "force.h"
#include "modify.h"
#include "pair_mliap.h"
#include "pair_curator.h"
#include "update.h"
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <map>
#include <unordered_map>

using namespace LAMMPS_NS;

namespace {

std::map<Pair *, std::map<std::string, double>> pair_uncertainties;

}

void LAMMPS_NS::clear_pair_uncertainties(Pair *pair)
{
  auto &values = pair_uncertainties[pair];
  for (auto &entry : values) entry.second = 0.0;
}

void LAMMPS_NS::set_pair_uncertainty(Pair *pair, const std::string &name, double value)
{
  pair_uncertainties[pair][name] = value;
}

double *LAMMPS_NS::get_pair_uncertainty_ptr(Pair *pair, const std::string &name)
{
  return &pair_uncertainties[pair][name];
}

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

  // Backward-compatible fallback for pair_curator, which does not expose Pair::extract().
  pair_curator = dynamic_cast<PairCurator *>(pair_ptr);
  if (pair_curator != NULL) return;

  if (dynamic_cast<PairMLIAP *>(pair_ptr) != NULL) {
    uncertainty_ptr = get_pair_uncertainty_ptr(pair_ptr, uncertainty_name);
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
  if (uncertainty_ptr != NULL) {
    scalar = *uncertainty_ptr;
  } else {
    scalar = pair_curator->get_uncertainty(uncertainty_name);
  }
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
