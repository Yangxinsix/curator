/* ----------------------------------------------------------------------
References:

   .. [#pair_nequip] https://github.com/mir-group/pair_nequip
   .. [#lammps] https://github.com/lammps/lammps

------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(painn,PairPAINN)

#else

#ifndef LMP_PAIR_PAINN_H
#define LMP_PAIR_PAINN_H

#include "pair.h"

#include <torch/torch.h>

namespace LAMMPS_NS {
    
class PairPAINN : public Pair {
 public:
  PairPAINN(class LAMMPS *);
  virtual ~PairPAINN();
  virtual void compute(int, int);
  void settings(int, char **);
  virtual void coeff(int, char **);
  virtual double init_one(int, int);
  virtual void init_style();
  void allocate();

  double cutoff;
  torch::jit::script::Module model;
  torch::Device device = torch::kCPU;

 protected:
  int * type_mapper;
  int debug_mode = 0;
  int ensemble = 0;

};

}

#endif
#endif
