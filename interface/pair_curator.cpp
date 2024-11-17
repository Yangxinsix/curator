/* ----------------------------------------------------------------------
References:

   .. [#pair_nequip] https://github.com/mir-group/pair_nequip
   .. [#lammps] https://github.com/lammps/lammps

------------------------------------------------------------------------- */

#include <pair_curator.h>
#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "potential_file_reader.h"
#include "tokenizer.h"

#include <unordered_map>
#include <string>
#include <cmath>
#include <cstring>
#include <numeric>
#include <cassert>
#include <iostream>
#include <sstream>
#include <string>
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/csrc/jit/runtime/graph_executor.h>


using namespace LAMMPS_NS;

PairCurator::PairCurator(LAMMPS *lmp) : Pair(lmp) {
  restartinfo = 0;
  manybody_flag = 1;
  compute_uncertainty = 0;
  debug_mode = 0;

  if(torch::cuda::is_available()){
    device = torch::kCUDA;
  }
  else {
    device = torch::kCPU;
  }

  if (comm->me == 0) {
    std::cout << "CURATOR is using device: " << device << std::endl;
  }

  if(const char* env_p = std::getenv("CURATOR_DEBUG")){
    if (comm->me == 0) {
      std::cout << "PairCurator is in DEBUG mode, since CURATOR_DEBUG is set in the environment" << std::endl;
    }
    debug_mode = 1;
  }
}

PairCurator::~PairCurator(){
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(type_mapper);
  }
}

void PairCurator::init_style(){
  if (atom->tag_enable == 0)
    error->all(FLERR,"Pair style curator requires atom IDs");

  // need a full neighbor list
  neighbor->add_request(this, NeighConst::REQ_FULL);

  // TODO: I think Newton should be off, enforce this.
  // The network should just directly compute the total forces
  // on the "real" atoms, with no need for reverse "communication".
  // May not matter, since f[j] will be 0 for the ghost atoms anyways.
  if (force->newton_pair == 1)
    error->all(FLERR,"Pair style curator requires newton pair off");
}

double PairCurator::init_one(int i, int j)
{
  return cutoff;
}

void PairCurator::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  memory->create(cutsq,n+1,n+1,"pair:cutsq");
  memory->create(type_mapper, n+1, "pair:type_mapper");

}

void PairCurator::settings(int narg, char **arg) {
  // "uncertainty" should be after "pair_style" in the input file if you want to calculate uncertainty.
  if (narg > 0) {
    if (strcmp(arg[0], "uncertainty") == 0) {
      compute_uncertainty = 1;
      uncertainties.clear();
      if (narg == 1) uncertainties["force_sd"] = 0.0;      // default is to extract force standard deviation
      else {
        for (int i = 1; i < narg; ++i) {
          uncertainties[std::string(arg[i])] = 0.0;
        }
      }
    }
    else {
      error->all(FLERR, "Illegal pair_style command: unknown keyword");
    }
  }
}

void PairCurator::coeff(int narg, char **arg) {

  if (!allocated)
    allocate();

  int ntypes = atom->ntypes;

  // Should be exactly 3 arguments following "pair_coeff" in the input file.
  if (narg != (3+ntypes))
    error->all(FLERR, "Incorrect args for pair coefficients");

  // Ensure I,J args are "* *".
  if (strcmp(arg[0], "*") != 0 || strcmp(arg[1], "*") != 0)
    error->all(FLERR, "Incorrect args for pair coefficients");

  for (int i = 1; i <= ntypes; i++)
    for (int j = i; j <= ntypes; j++)
      setflag[i][j] = 0;

  // Initiate type mapper
  for (int i = 1; i<= ntypes; i++){
      type_mapper[i] = -1;
  }

  if (comm->me == 0) {
    std::cout << "Loading model from " << arg[2] << std::endl;
  }

  
  std::unordered_map<std::string, std::string> metadata = {
    {"cutoff", ""},
  };
  model = torch::jit::load(std::string(arg[2]), device, metadata);
  model.eval();

  cutoff = std::stod(metadata["cutoff"]);

  // match lammps types to atomic numbers
  int counter = 1;
  for (int i = 3; i < narg; i++){
      type_mapper[counter] = std::stoi(arg[i]);
      counter++;
  }
  
  if(debug_mode && comm->me == 0){
    std::cout << "cutoff" << cutoff << "\n";
    for (int i = 0; i <= ntypes+1; i++){
        std::cout << type_mapper[i] << "\n";
    }
  }

  // set setflag i,j for type pairs where both are mapped to elements
  for (int i = 1; i <= ntypes; i++)
    for (int j = i; j <= ntypes; j++)
        if ((type_mapper[i] >= 0) && (type_mapper[j] >= 0))
            setflag[i][j] = 1;

}

// Force and energy computation
void PairCurator::compute(int eflag, int vflag){
  ev_init(eflag, vflag);

  // Get info from lammps:
  // Atom positions, including ghost atoms
  double **x = atom->x;
  // Atom forces
  double **f = atom->f;
  // Whether Newton is on (i.e. reverse "communication" of forces on ghost atoms).
  // Should probably be off.
  if (force->newton_pair==1)
    error->all(FLERR, "Pair style curator requires 'newton off'");

  assert(list->inum==atom->nlocal); // This should be true, if my understanding is correct
  // Total number of atoms
  int ntotal = list->inum + list->gnum;;

  // Total number of bonds (sum of number of neighbors)
  int nedges = std::accumulate(list->numneigh, list->numneigh+ntotal, 0);
  torch::Tensor tag2type_tensor = torch::zeros({atom->nlocal}, torch::TensorOptions().dtype(torch::kInt64));
  auto tag2type = tag2type_tensor.accessor<long, 1>();

  // Inverse mapping from tag to "real" atom index
  std::vector<int> tag2i(list->inum);

  // Loop over real atoms to store tags, types and positions
  for(int ii = 0; ii < list->inum; ii++){
    int i = list->ilist[ii];
    int itag = atom->tag[i];
    int itype = atom->type[i];  // type is 1-based

    // Inverse mapping from tag to x/f atom index
    tag2i[itag-1] = i; // tag is probably 1-based
    tag2type[itag-1] = type_mapper[itype];
  }

  // Loop over atoms and neighbors,
  // store edges and edge_diff
  // ii follows the order of the neighbor lists,
  // i follows the order of x, f, etc.
  long edges[nedges][2];
  double edge_diff[nedges][3];
  int edge_counter = 0;
  if (debug_mode) {
    std::cout << "num_atoms = " << atom->nlocal << std::endl;
    std::cout << "nedges = " << nedges << std::endl;
    std::cout << "elems = " << tag2type_tensor << std::endl;
  }
  if (debug_mode) printf("curator edges: i j xi[:] xj[:]\n");
  for(int ii = 0; ii < list->inum; ii++){
    int i = list->ilist[ii];
    int itag = atom->tag[i];   // atom tag is 1-based
    int itype = atom->type[i];
    if (debug_mode) printf("i_index: %d type: %d num_neigh: %d\n", itag, itype, list->numneigh[i]);

    for(int jj = 0; jj < list->numneigh[ii]; jj++){
      int j = list->firstneigh[ii][jj];
      j &= NEIGHMASK;
      int jtag = atom->tag[j];
      int jtype = atom->type[j];

      double dx = x[j][0] - x[i][0];
      double dy = x[j][1] - x[i][1];
      double dz = x[j][2] - x[i][2];

      domain->minimum_image(dx, dy, dz);
      double rsq = dx*dx + dy*dy + dz*dz;
      if (rsq < cutoff*cutoff){
          // TODO: double check order
          edges[edge_counter][0] = itag - 1; // tag is probably 1-based
          edges[edge_counter][1] = jtag - 1; // tag is probably 1-based
          edge_diff[edge_counter][0] = dx;
          edge_diff[edge_counter][1] = dy;
          edge_diff[edge_counter][2] = dz;
          edge_counter++;

          if (debug_mode){
              printf("%d %d %.10g %.10g %.10g %.10g\n", itag-1, jtag-1,
                dx,dy,dz,sqrt(rsq));
          }

      }
    }
  }
  if (debug_mode) printf("end curator edges\n");

  // shorten the list before sending to nequip
  torch::Tensor edges_tensor = torch::from_blob(edges, {edge_counter, 2}, torch::TensorOptions().dtype(torch::kInt64));
  torch::Tensor edge_diff_tensor = torch::from_blob(edge_diff, {edge_counter, 3}, torch::TensorOptions().dtype(torch::kFloat64));
  edge_diff_tensor = edge_diff_tensor.to(torch::kFloat32);
 
  // define curator n_atoms input
  torch::Tensor n_atoms_tensor = torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt64));
  n_atoms_tensor[0] = atom->nlocal;
  torch::Tensor n_pairs_tensor = torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt64));
  n_pairs_tensor[0] = edge_counter;

  c10::Dict<std::string, torch::Tensor> input;
  input.insert("n_atoms", n_atoms_tensor.to(device));
  input.insert("_n_pairs", n_pairs_tensor.to(device));
  input.insert("_edge_index" , edges_tensor.to(device));
  input.insert("_edge_difference", edge_diff_tensor.to(device));
  input.insert("atomic_numbers", tag2type_tensor.to(device));

  if(debug_mode){
    std::cout << "curator model input:\n";
    std::cout << "num_atoms:\n" << n_atoms_tensor << "\n";
    std::cout << "num_pairs:\n" << n_pairs_tensor << "\n";
    std::cout << "edge_index:\n" << edges_tensor << "\n";
    std::cout << "edge_difference:\n" << edge_diff_tensor<< "\n";
    std::cout << "atomic_numbers:\n" << tag2type_tensor << "\n";
  }

  std::vector<torch::IValue> input_vector(1, input);

  auto output = model.forward(input_vector).toGenericDict();
  
  // get forces
  torch::Tensor forces_tensor = output.at("forces").toTensor().cpu();
  auto forces = forces_tensor.accessor<float, 2>();

  // get energy
  torch::Tensor total_energy_tensor = output.at("energy").toTensor().cpu();
  // store the total energy where LAMMPS wants it
  eng_vdwl = total_energy_tensor.data_ptr<float>()[0];

  // get virial
  auto it = output.find("virial");
  if (it != output.end()) {
    torch::Tensor virial_tensor = output.at("virial").toTensor().squeeze().cpu();
    auto pred_virials = virial_tensor.accessor<float, 1>();
    virial[0] = pred_virials[0];
    virial[1] = pred_virials[1];
    virial[2] = pred_virials[2];
    virial[3] = pred_virials[3];
    virial[4] = pred_virials[4];
    virial[5] = pred_virials[5];
  }

  // Get uncertainties
  if (compute_uncertainty) {
    for (auto& pair : uncertainties) {
      const std::string &name = pair.first;
      auto it = output.find(name);
      if (it != output.end()) {
        torch::Tensor uncertainty_tensor = output.at(name).toTensor().cpu();
        pair.second = uncertainty_tensor.item<float>(); // Update the uncertainty value
      } else {
        std::string error_msg = "Uncertainty key '" + name + "' not found in model output.";
        error->all(FLERR, error_msg.c_str());
      }
    }
  }

  if(debug_mode){
    std::cout << "curator model output:\n";
    std::cout << "forces: " << forces_tensor << "\n";
    std::cout << "energy: " << total_energy_tensor << "\n";
    if (compute_uncertainty) {
      for (const auto& pair : uncertainties) {
        std::cout << "Key: " << pair.first << ", Value: " << pair.second << std::endl;
      }
    }
  }
  
  // Write forces and per-atom energies (0-based tags here)
  for(int itag = 0; itag < list->inum; itag++){
    int i = tag2i[itag];
    f[i][0] = forces[itag][0];
    f[i][1] = forces[itag][1];
    f[i][2] = forces[itag][2];
  }
}

double PairCurator::get_uncertainty(const std::string &name) const {
  auto it = uncertainties.find(name);
  if (it != uncertainties.end()) {
    return it->second;
  } else {
    std::string error_msg = "Uncertainty '" + name + "' not found in PairCurator.";
    error->all(FLERR, error_msg.c_str());
    return 0.0; // This line will not be reached due to error->all()
  }
}