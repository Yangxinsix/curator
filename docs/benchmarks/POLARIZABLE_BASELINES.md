# Polarizable baselines: implementation boundaries

## Extended LES

The Curator implementation and its validation are documented in
[LATENT_EWALD.md](LATENT_EWALD.md). PolarizableLatentEwaldEnergy invokes the
official LES library for permanent dipoles and isotropic polarizability with one
induced-response step. The current wrapper does not claim quadrupoles, a fully
self-consistent iterative polarization solver, or exact reproduction of the
authors' complete training recipe. Its bounded polarizability scale is a Curator
hyperparameter, not a stability theorem.

This supplies a directly trainable PyTorch physical baseline in the same Curator
EF training workflow. The LES source remains an isolated optional dependency;
the official source and data licenses must remain attached to their artifacts.

Official sources:
- https://github.com/ChengUCB/les
- https://github.com/ChengUCB/extended_les_fit
- https://arxiv.org/abs/2605.05746

## ReaxNet / PQEq

ReaxNet's official implementation is JAX, with JAX-MD, e3nn_jax, Flax, Jraph and
Optax; the code is GPL-3.0. Its physical model includes polarizable core/shell
charge response. A point-charge QEq module with a changed hardness does not
reproduce that model.

No fake Torch adapter or simplified model labelled ReaxNet has been added here.
Converting JAX energy/forces to NumPy and then Torch would break the gradients to
the trainable model. An inference-only wrapper would not meet our Curator
energy/force-training requirement. A first-derivative custom bridge alone also
would not establish the mixed derivatives needed to learn from force losses.

Two concrete integration routes are possible:
1. Keep the official JAX model in an isolated runtime and evaluate its published
   checkpoints using our frozen structures and unit conventions. This yields an
   external reference, not a Curator-trainable matched-backbone comparison.
2. Independently implement the documented core/shell functional and constrained
   solver in PyTorch, including force differentiation through polarization, and
   verify energies, forces, response and parameter gradients against official
   ReaxNet on shared fixtures before calling it a PQEq reproduction. This is a
   distinct implementation effort, not an adapter-only task.

The present delivery chooses the official PyTorch extended-LES route for a
trainable polarizable baseline and records PQEq as not yet implemented. The
JAX dependency stack has not been installed into the Curator image.

Primary sources:
- Official code/license and environment: https://github.com/reaxnet/reaxnet
- Paper: https://www.nature.com/articles/s41467-025-65496-3
- Official physical-term example: https://github.com/reaxnet/reaxnet/blob/main/examples/non_bond.ipynb

