#include "compute_uncertainty_atom.h"

#include "atom.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "pair_curator.h"
#include "pair_mliap.h"
#include "update.h"

#include <cstdlib>
#include <iostream>
#include <map>

using namespace LAMMPS_NS;

namespace {

std::map<Pair *, std::map<std::string, std::vector<double>>> pair_uncertainty_arrays;

}

void LAMMPS_NS::clear_pair_uncertainty_arrays(Pair *pair)
{
  pair_uncertainty_arrays[pair].clear();
}

void LAMMPS_NS::set_pair_uncertainty_array(
  Pair *pair,
  const std::string &name,
  const double *values,
  int nvalues
)
{
  auto &target = pair_uncertainty_arrays[pair][name];
  if (values == nullptr || nvalues <= 0) {
    target.clear();
    return;
  }
  target.assign(values, values + nvalues);
}

double *LAMMPS_NS::get_pair_uncertainty_array_ptr(Pair *pair, const std::string &name, int &nvalues)
{
  nvalues = 0;
  auto pair_it = pair_uncertainty_arrays.find(pair);
  if (pair_it == pair_uncertainty_arrays.end()) return nullptr;

  auto value_it = pair_it->second.find(name);
  if (value_it == pair_it->second.end()) return nullptr;
  if (value_it->second.empty()) return nullptr;

  nvalues = static_cast<int>(value_it->second.size());
  return value_it->second.data();
}

ComputeUncertaintyAtom::ComputeUncertaintyAtom(LAMMPS *lmp, int narg, char **arg)
  : Compute(lmp, narg, arg)
{
  if (narg != 4)
    error->all(FLERR, "Illegal compute uncertainty/atom command");

  uncertainty_name = std::string(arg[3]);
  peratom_flag = 1;
  size_peratom_cols = 0;

  debug_mode = 0;
  nmax = 0;
  pair_ptr = nullptr;
  uncertainty_array = nullptr;

  if (const char *env_p = std::getenv("CURATOR_DEBUG")) {
    debug_mode = 1;
  }
}

ComputeUncertaintyAtom::~ComputeUncertaintyAtom()
{
  memory->destroy(uncertainty_array);
}

void ComputeUncertaintyAtom::init()
{
  pair_ptr = force->pair;
  if (pair_ptr == nullptr)
    error->all(FLERR, "Compute uncertainty/atom requires a pair style to be defined");

  if (dynamic_cast<PairMLIAP *>(pair_ptr) == nullptr &&
      dynamic_cast<PairCurator *>(pair_ptr) == nullptr) {
    error->all(
      FLERR,
      "Compute uncertainty/atom currently requires pair_style mliap or curator with the CURATOR bridge"
    );
  }
}

void ComputeUncertaintyAtom::compute_peratom()
{
  invoked_peratom = update->ntimestep;

  if (atom->nmax > nmax) {
    memory->destroy(uncertainty_array);
    nmax = atom->nmax;
    memory->create(uncertainty_array, nmax, "compute/uncertainty/atom:uncertainty_array");
    vector_atom = uncertainty_array;
  }

  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int nvalues = 0;
  int columns = 0;
  double *values = static_cast<double *>(pair_ptr->extract_peratom(uncertainty_name.c_str(), columns));

  if (values != nullptr && columns != 0) {
    error->all(
      FLERR,
      "Compute uncertainty/atom expects a per-atom vector, not a per-atom array"
    );
  }

  if (values != nullptr) {
    for (int i = 0; i < nlocal; i++) {
      uncertainty_array[i] = (mask[i] & groupbit) ? values[i] : 0.0;
    }
    if (debug_mode) {
      std::cout << "Key: " << uncertainty_name << ", Nlocal: " << nlocal << std::endl;
    }
    return;
  }

  // Backward-compatible fallback for pair_curator, which still stores per-atom
  // uncertainty arrays outside Pair::extract_peratom().
  values = get_pair_uncertainty_array_ptr(pair_ptr, uncertainty_name, nvalues);

  if (values == nullptr) {
    for (int i = 0; i < nlocal; i++) uncertainty_array[i] = 0.0;
    return;
  }

  if (nvalues < nlocal) {
    error->all(
      FLERR,
      "Compute uncertainty/atom received fewer uncertainty values than local atoms"
    );
  }

  for (int i = 0; i < nlocal; i++) {
    uncertainty_array[i] = (mask[i] & groupbit) ? values[i] : 0.0;
  }

  if (debug_mode) {
    std::cout << "Key: " << uncertainty_name << ", Nlocal: " << nlocal << std::endl;
  }
}

double ComputeUncertaintyAtom::memory_usage()
{
  return static_cast<double>(nmax) * sizeof(double);
}
