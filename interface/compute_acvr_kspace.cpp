/* ----------------------------------------------------------------------
   ACVR point-charge Ewald/PPPM site-potential operator for LAMMPS.

   The returned per-atom vector is K q in energy/charge units.  It combines
   the ELECTRODE KSpace reciprocal action with the point-charge real-space,
   self, boundary-correction, and optional uniform-background terms.
------------------------------------------------------------------------- */

#include "compute_acvr_kspace.h"

#include "ELECTRODE/electrode_kspace.h"
#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "kspace.h"
#include "math_const.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "pair.h"
#include "update.h"
#include "utils.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace MathConst;

ComputeACVRKSpace::ComputeACVRKSpace(LAMMPS *lmp, int narg, char **arg) :
    Compute(lmp, narg, arg), source_group(nullptr), potential(nullptr), list(nullptr),
    electrode_kspace(nullptr)
{
  if (narg != 4) error->all(FLERR, "Illegal compute acvr/kspace command");

  source_group = utils::strdup(arg[3]);
  cutoff = cutoff_sq = 0.0;

  peratom_flag = 1;
  size_peratom_cols = 0;
  scalar_flag = 1;
  extscalar = 1;
  nmax = 0;
}

ComputeACVRKSpace::~ComputeACVRKSpace()
{
  delete[] source_group;
  memory->destroy(potential);
}

void ComputeACVRKSpace::init()
{
  if (!atom->q_flag) error->all(FLERR, "Compute acvr/kspace requires atom attribute q");
  if (force->kspace == nullptr)
    error->all(FLERR, "Compute acvr/kspace requires a KSpace style");
  electrode_kspace = dynamic_cast<ElectrodeKSpace *>(force->kspace);
  if (electrode_kspace == nullptr)
    error->all(FLERR, "Compute acvr/kspace requires ewald/electrode or pppm/electrode");

  if (force->pair == nullptr)
    error->all(FLERR, "Compute acvr/kspace requires a long-range Coulomb pair style");
  int dimension = 0;
  auto *pair_cutoff = static_cast<double *>(force->pair->extract("cut_coul", dimension));
  if (pair_cutoff == nullptr || *pair_cutoff <= 0.0)
    error->all(FLERR, "Compute acvr/kspace pair style does not expose cut_coul");
  cutoff = *pair_cutoff;
  cutoff_sq = cutoff * cutoff;

  source_groupbit = group->get_bitmask_by_id(FLERR, source_group, "compute acvr/kspace");
  auto *request = neighbor->add_request(
      this, NeighConst::REQ_FULL | NeighConst::REQ_OCCASIONAL);
  request->set_cutoff(cutoff);
}

void ComputeACVRKSpace::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

void ComputeACVRKSpace::compute_peratom()
{
  invoked_peratom = update->ntimestep;
  if (atom->nmax > nmax) {
    memory->destroy(potential);
    nmax = atom->nmax;
    memory->create(potential, nmax, "compute/acvr/kspace:potential");
    vector_atom = potential;
  }
  memset(potential, 0, nmax * sizeof(double));

  neighbor->build_one(list);
  electrode_kspace->compute_vector(potential, groupbit, source_groupbit, false);
  electrode_kspace->compute_vector_corr(potential, groupbit, source_groupbit, false);
  add_real_space();
  add_self_and_background();

  const double conversion = force->qqrd2e;
  const int nlocal = atom->nlocal;
  int *mask = atom->mask;
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit)
      potential[i] *= conversion;
    else
      potential[i] = 0.0;
  }
}

void ComputeACVRKSpace::add_real_space()
{
  double **x = atom->x;
  double *q = atom->q;
  int *mask = atom->mask;
  double *special_coul = force->special_coul;
  const double alpha = force->kspace->g_ewald;

  const int inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;
  for (int ii = 0; ii < inum; ii++) {
    const int i = ilist[ii];
    if (!(mask[i] & groupbit)) continue;
    int *jlist = firstneigh[i];
    const int jnum = numneigh[i];
    for (int jj = 0; jj < jnum; jj++) {
      int encoded_j = jlist[jj];
      const double factor_coul = special_coul[sbmask(encoded_j)];
      const int j = encoded_j & NEIGHMASK;
      if (!(mask[j] & source_groupbit)) continue;
      const double dx = x[i][0] - x[j][0];
      const double dy = x[i][1] - x[j][1];
      const double dz = x[i][2] - x[j][2];
      const double rsq = dx * dx + dy * dy + dz * dz;
      if (rsq <= 0.0 || rsq >= cutoff_sq) continue;
      const double r = std::sqrt(rsq);
      const double split = std::erfc(alpha * r) + factor_coul - 1.0;
      potential[i] += q[j] * split / r;
    }
  }
}

void ComputeACVRKSpace::add_self_and_background()
{
  double *q = atom->q;
  int *mask = atom->mask;
  const int nlocal = atom->nlocal;
  const double alpha = force->kspace->g_ewald;
  const double self = -2.0 * alpha / MY_PIS;

  double source_charge_local = 0.0;
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & source_groupbit) source_charge_local += q[i];
    if ((mask[i] & groupbit) && (mask[i] & source_groupbit))
      potential[i] += self * q[i];
  }
  double source_charge = 0.0;
  MPI_Allreduce(&source_charge_local, &source_charge, 1, MPI_DOUBLE, MPI_SUM, world);
  const double volume = domain->xprd * force->kspace->wire_volfactor
      * domain->yprd * force->kspace->wire_volfactor
      * domain->zprd * force->kspace->slab_volfactor;
  const double background = -MY_PI * source_charge / (alpha * alpha * volume);
  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) potential[i] += background;
}

double ComputeACVRKSpace::compute_scalar()
{
  if (invoked_peratom != update->ntimestep) compute_peratom();
  double local = 0.0;
  double *q = atom->q;
  int *mask = atom->mask;
  for (int i = 0; i < atom->nlocal; i++)
    if (mask[i] & groupbit) local += q[i] * potential[i];
  double global = 0.0;
  MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, world);
  scalar = (groupbit == source_groupbit) ? 0.5 * global : global;
  invoked_scalar = update->ntimestep;
  return scalar;
}

double ComputeACVRKSpace::memory_usage()
{
  return static_cast<double>(nmax) * sizeof(double);
}
