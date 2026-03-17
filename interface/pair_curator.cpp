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

#include <cassert>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>
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

  int nlocal = atom->nlocal;
  int nall = atom->nlocal + atom->nghost;
  tagint *tag = atom->tag;
  constexpr double zero_tol = 1e-20;
  std::unordered_map<tagint, int> owned_index_by_tag;
  owned_index_by_tag.reserve(nlocal);
  for (int i = 0; i < nlocal; i++) owned_index_by_tag[tag[i]] = i;

  // Total number of bonds (sum of number of neighbors)
  int nedges = std::accumulate(list->numneigh, list->numneigh + list->inum, 0);
  torch::Tensor atomic_numbers_tensor =
      torch::zeros({nlocal}, torch::TensorOptions().dtype(torch::kInt64));
  auto atomic_numbers = atomic_numbers_tensor.accessor<long, 1>();

  for (int i = 0; i < nlocal; i++) {
    int itype = atom->type[i];  // type is 1-based
    atomic_numbers[i] = type_mapper[itype];
  }

  // Loop over atoms and neighbors,
  // store edges and edge_diff
  // ii follows the order of the neighbor lists,
  // i follows the order of x, f, etc.
  std::vector<int64_t> edges(2 * nedges);
  std::vector<double> edge_diff(3 * nedges);
  int edge_counter = 0;
  int skipped_zero_edges = 0;
  int skipped_self_image_edges = 0;
  int missing_owned_tag_edges = 0;
  if (debug_mode) {
    std::cout << "num_atoms = " << nlocal << std::endl;
    std::cout << "num_lammps_atoms = " << nall << std::endl;
    std::cout << "nedges = " << nedges << std::endl;
    std::cout << "elems = " << atomic_numbers_tensor << std::endl;
  }
  if (debug_mode) printf("curator edges: i j xi[:] xj[:]\n");
  for(int ii = 0; ii < list->inum; ii++){
    int i = list->ilist[ii];
    int itype = atom->type[i];
    if (debug_mode) printf("i_index: %d type: %d num_neigh: %d\n", i, itype, list->numneigh[ii]);

    for(int jj = 0; jj < list->numneigh[ii]; jj++){
      int j = list->firstneigh[ii][jj];
      j &= NEIGHMASK;

      double dx = x[j][0] - x[i][0];
      double dy = x[j][1] - x[i][1];
      double dz = x[j][2] - x[i][2];
      double rsq = dx*dx + dy*dy + dz*dz;
      if (rsq <= zero_tol) {
          skipped_zero_edges++;
          if (tag != nullptr && tag[i] == tag[j]) skipped_self_image_edges++;
          continue;
      }
      if (rsq < cutoff*cutoff){
          auto owned_j = owned_index_by_tag.find(tag[j]);
          if (owned_j == owned_index_by_tag.end()) {
              missing_owned_tag_edges++;
              continue;
          }
          edges[2 * edge_counter] = i;
          edges[2 * edge_counter + 1] = owned_j->second;
          edge_diff[3 * edge_counter] = dx;
          edge_diff[3 * edge_counter + 1] = dy;
          edge_diff[3 * edge_counter + 2] = dz;
          edge_counter++;

          if (debug_mode){
              printf("%d %d %.10g %.10g %.10g %.10g\n", i, j,
                dx,dy,dz,sqrt(rsq));
          }

      }
    }
  }
  if (debug_mode) {
    std::cout << "skipped_zero_edges = " << skipped_zero_edges << std::endl;
    std::cout << "skipped_self_image_edges = " << skipped_self_image_edges << std::endl;
    std::cout << "missing_owned_tag_edges = " << missing_owned_tag_edges << std::endl;
  }
  if (missing_owned_tag_edges > 0)
    error->all(FLERR, "Pair style curator found ghost neighbors whose tags are not owned locally; current pair_curator graph export only supports local-owned node indexing");
  if (debug_mode) printf("end curator edges\n");

  // shorten the list before sending to nequip
  torch::Tensor edges_tensor;
  torch::Tensor edge_diff_tensor;
  if (edge_counter > 0) {
    edges_tensor = torch::from_blob(
        edges.data(), {edge_counter, 2}, torch::TensorOptions().dtype(torch::kInt64));
    edge_diff_tensor = torch::from_blob(
        edge_diff.data(), {edge_counter, 3}, torch::TensorOptions().dtype(torch::kFloat64));
  } else {
    edges_tensor = torch::empty({0, 2}, torch::TensorOptions().dtype(torch::kInt64));
    edge_diff_tensor = torch::empty({0, 3}, torch::TensorOptions().dtype(torch::kFloat64));
  }
  edge_diff_tensor = edge_diff_tensor.to(torch::kFloat32);
 
  // define curator n_atoms input
  torch::Tensor n_atoms_tensor = torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt64));
  n_atoms_tensor[0] = nlocal;
  torch::Tensor n_pairs_tensor = torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt64));
  n_pairs_tensor[0] = edge_counter;

  c10::Dict<std::string, torch::Tensor> input;
  input.insert("n_atoms", n_atoms_tensor.to(device));
  input.insert("_n_pairs", n_pairs_tensor.to(device));
  input.insert("_edge_index" , edges_tensor.to(device));
  input.insert("_edge_difference", edge_diff_tensor.to(device));
  input.insert("atomic_numbers", atomic_numbers_tensor.to(device));

  if(debug_mode){
    std::cout << "curator model input:\n";
    std::cout << "num_atoms:\n" << n_atoms_tensor << "\n";
    std::cout << "num_pairs:\n" << n_pairs_tensor << "\n";
    std::cout << "edge_index:\n" << edges_tensor << "\n";
    std::cout << "edge_difference:\n" << edge_diff_tensor<< "\n";
    std::cout << "atomic_numbers:\n" << atomic_numbers_tensor << "\n";
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
    torch::Tensor virial_tensor = output.at("virial").toTensor().cpu();
    torch::Tensor virial_local;
    if (virial_tensor.dim() == 1) virial_local = virial_tensor;
    else virial_local = virial_tensor.reshape({-1, virial_tensor.size(-1)})[0];
    auto pred_virials = virial_local.accessor<float, 1>();
    // curator uses Voigt notation for virial tensors: xx,yy,zz,yz,xz,xy. lammps: xx,yy,zz,xy,xz,yz
    virial[0] = pred_virials[0];
    virial[1] = pred_virials[1];
    virial[2] = pred_virials[2];
    virial[3] = pred_virials[5];
    virial[4] = pred_virials[4];
    virial[5] = pred_virials[3];
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
  
  // Write forces for local atoms
  for (int i = 0; i < nlocal; i++) {
    f[i][0] = forces[i][0];
    f[i][1] = forces[i][1];
    f[i][2] = forces[i][2];
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

void *PairCurator::extract(const char *name, int &dim)
{
  dim = 0;
  auto it = uncertainties.find(std::string(name));
  if (it == uncertainties.end()) return nullptr;
  return static_cast<void *>(&it->second);
}
