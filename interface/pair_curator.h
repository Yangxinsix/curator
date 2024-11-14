/* ----------------------------------------------------------------------
References:

   .. [#pair_nequip] https://github.com/mir-group/pair_nequip
   .. [#lammps] https://github.com/lammps/lammps

------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(curator,PairCurator)

#else

#ifndef LMP_PAIR_CURATOR_H
#define LMP_PAIR_CURATOR_H

#include "pair.h"

#include <torch/torch.h>

namespace LAMMPS_NS {
    
class PairCurator : public Pair {
 public:
  PairCurator(class LAMMPS *);
  virtual ~PairCurator();
  virtual void compute(int, int);
  void settings(int, char **);
  virtual void coeff(int, char **);
  virtual double init_one(int, int);
  virtual void init_style();
  void allocate();
  double cutoff;
  torch::jit::script::Module model;
  torch::Device device = torch::kCPU;
  // Method to access uncertainties
  double get_uncertainty(const std::string &name) const;
  // Uncertainty storage
  std::unordered_map<std::string, double> uncertainties;

 private:
  int debug_mode;
  int compute_uncertainty;

 protected:
  int * type_mapper;

};

}

#endif
#endif
