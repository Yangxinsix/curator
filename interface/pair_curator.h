/* ----------------------------------------------------------------------
References:

   .. [#pair_nequip] https://github.com/mir-group/pair_nequip
   .. [#lammps] https://github.com/lammps/lammps

------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(painn,PairCURATOR)

#else

#ifndef LMP_PAIR_CURATOR_H
#define LMP_PAIR_CURATOR_H

#include "pair.h"

#include <torch/torch.h>

namespace LAMMPS_NS {
    
class PairCURATOR : public Pair {
 public:
  PairCURATOR(class LAMMPS *);
  virtual ~PairCURATOR();
  virtual void compute(int, int);
  void settings(int, char **);
  virtual void coeff(int, char **);
  virtual double init_one(int, int);
  virtual void init_style();
  void allocate();

  double cutoff;
  torch::jit::script::Module model;
  torch::Device device = torch::kCPU;

  // uncertainty information
  double uncertainty_scalar;

 protected:
  int * type_mapper;
  int debug_mode = 0;
  int ensemble = 0;

};

}

#endif
#endif
