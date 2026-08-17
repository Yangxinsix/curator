/* -*- c++ -*- ----------------------------------------------------------
   ACVR point-charge Ewald/PPPM site-potential operator for LAMMPS.
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(acvr/kspace,ComputeACVRKSpace);
// clang-format on
#else

#ifndef LMP_COMPUTE_ACVR_KSPACE_H
#define LMP_COMPUTE_ACVR_KSPACE_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeACVRKSpace : public Compute {
 public:
  ComputeACVRKSpace(class LAMMPS *, int, char **);
  ~ComputeACVRKSpace() override;

  void init() override;
  void init_list(int, class NeighList *) override;
  void compute_peratom() override;
  double compute_scalar() override;
  double memory_usage() override;

 private:
  char *source_group;
  int source_groupbit;
  int nmax;
  double cutoff, cutoff_sq;
  double *potential;
  class NeighList *list;
  class ElectrodeKSpace *electrode_kspace;

  void add_real_space();
  void add_self_and_background();
};

}    // namespace LAMMPS_NS

#endif
#endif
