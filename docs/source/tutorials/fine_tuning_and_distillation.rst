.. _fine-tuning-and-distillation:

Fine-tuning and Knowledge Distillation
======================================

CURATOR can adapt an existing potential instead of training every model from
scratch. Fine-tuning controls which student parameters are updated, while
knowledge distillation controls which signals the student learns from. The two
features are independent and can also be combined.

This guide covers:

* full, head-only, and LoRA fine-tuning;
* the ``model``, ``weights``, and ``resume`` checkpoint-loading modes;
* external pretrained MACE, NequIP, and MatGL models;
* offline energy/force distillation;
* sampled, projected, and dynamic projected-Hessian distillation; and
* generation, validation, reuse, and recovery of teacher-label databases.

All examples are user configuration files passed to:

.. code-block:: bash

   curator-train cfg=<config.yaml>


Fine-tuning
-----------

Fine-tuning policy and checkpoint loading are separate decisions:

* ``finetune`` selects which parameter groups the optimizer updates;
* ``model_path`` selects the source model and how it is loaded.

Fine-tuning policies
~~~~~~~~~~~~~~~~~~~~

CURATOR provides three policies in ``curator/configs/finetune/``.

.. list-table:: Fine-tuning policies
   :header-rows: 1
   :widths: 18 42 40

   * - Policy
     - Optimization behavior
     - Typical use
   * - ``full``
     - All parameters use their configured optimizer groups.
     - Maximum flexibility when enough target-domain data are available.
   * - ``head_only``
     - The base learning rate is set to zero and the representation readout
       receives its own non-zero learning rate.
     - Conservative adaptation when the source representation already covers
       the target chemistry.
   * - ``lora``
     - Supported equivariant operators receive trainable low-rank updates.
       Wrapped base weights are frozen by default.
     - Parameter-efficient adaptation of a large pretrained potential.

Select a policy through the defaults list:

.. code-block:: yaml

   defaults:
     - finetune: lora

The default is ``full``.

Checkpoint-loading modes
~~~~~~~~~~~~~~~~~~~~~~~~

``model_path`` accepts a path string or a mapping with ``path``, ``mode``, and
an optional ordered ``transform`` list.

.. list-table:: ``model_path.mode``
   :header-rows: 1
   :widths: 18 42 40

   * - Mode
     - What CURATOR loads
     - Use it for
   * - ``model``
     - The architecture and weights stored by the source model.
     - Adapting the source model as-is or loading an external pretrained model.
   * - ``weights``
     - A model built from the current config, followed by non-strict loading of
       compatible source tensors.
     - Changing the configured head, domain layout, or another compatible
       part of the model while retaining reusable weights.
   * - ``resume``
     - A CURATOR Lightning checkpoint, including optimizer and scheduler state.
     - Continuing an interrupted training run exactly where it stopped.

``resume`` is continuation, not fresh fine-tuning. External model specs do not
support ``resume``.

Full and head-only fine-tuning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A full fine-tuning config can be as small as:

.. code-block:: yaml

   defaults:
     - finetune: full

   model_path:
     path: /path/to/pretrained.ckpt
     mode: model

   data:
     datapath: /path/to/target-domain.traj

   run_path: runs/finetune-full

For readout-only adaptation, replace ``full`` with ``head_only``. Its packaged
preset uses a readout learning rate of ``5e-4``. Override it when necessary:

.. code-block:: yaml

   defaults:
     - finetune: head_only

   task:
     optimizer_groups:
       readout:
         lr: 1.0e-4
         weight_decay: 0.0

LoRA fine-tuning
~~~~~~~~~~~~~~~~

The LoRA preset defaults to rank 16, alpha 16, and frozen wrapped base weights.

.. code-block:: yaml

   defaults:
     - finetune: lora

   model_path:
     path: /path/to/pretrained.ckpt
     mode: weights

   data:
     datapath: /path/to/target-domain.traj

   wrapper:
     lora_rank: 16
     lora_alpha: 16.0
     lora_freeze_base: true
     # Optional: apply LoRA only to selected semantic construction groups.
     # lora_target_groups: [interactions, products]

   run_path: runs/finetune-lora

The ``e3nn``, ``cueq``, and ``oeq`` wrapper backends can be combined with the
LoRA adapter. For example:

.. code-block:: yaml

   wrapper:
     backend: cueq
     adapter: lora
     lora_rank: 8
     lora_alpha: 8.0
     lora_freeze_base: true

.. important::

   ``lora_freeze_base: true`` freezes the base weights inside LoRA-wrapped
   modules. Modules that are not wrapped retain their normal optimizer
   settings. Use ``lora_target_groups`` and ``task.optimizer_groups`` when a
   strict trainable-parameter boundary is required.

External pretrained models
~~~~~~~~~~~~~~~~~~~~~~~~~~

The training path accepts pretrained specs for MACE, NequIP, and MatGL. The
prefix identifies the adapter and the remaining value identifies a local file
or model resource.

.. code-block:: yaml

   # MACE model file
   model_path:
     path: mace:/path/to/foundation.model?head=0
     mode: model

.. code-block:: yaml

   # NequIP compiled package
   model_path:
     path: nequip:/path/to/model.nequip.zip
     mode: model

.. code-block:: yaml

   # Named MatGL model
   model_path:
     path: matgl:M3GNet
     mode: model

Use ``mode: model`` to retain the external architecture. ``mode: weights`` is
appropriate only when the model built by the CURATOR config is structurally
compatible with the source tensors.


Knowledge distillation
----------------------

Distillation trains a student against predictions from a stronger teacher
checkpoint or ensemble. CURATOR supports online teacher inference at the loss
level, but the recommended training path is offline distillation.

Offline distillation runs the teacher once, stores derived labels in SQLite,
and then trains only the student:

.. code-block:: text

   teacher checkpoint(s) -> one-time inference -> validated SQLite labels
                                                        |
   reference structures --------------------------------+-> student training
                                                        |
                                reused across epochs and runs

This avoids keeping the teacher on the training GPU, avoids repeating teacher
inference every epoch, and leaves the source trajectory unchanged.

Energy and force distillation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``energy_force_distill`` combines four objectives:

* energy against the reference label;
* forces against the reference label;
* energy against the teacher prediction; and
* forces against the teacher prediction.

A complete configuration is:

.. code-block:: yaml

   defaults:
     - task/outputs: energy_force_distill
     - task/distill: offline

   data:
     datapath: /path/to/reference.traj

   task:
     distill:
       teacher_model_path: /path/to/teacher.ckpt
       label_scope: dataset
     loss_weights:
       energy: 1.0
       forces: 100.0
       energy_distill: 1.0
       forces_distill: 100.0

   run_path: runs/distill-energy-force

On the first run, the default teacher-label path is:

.. code-block:: text

   runs/distill-energy-force/distill_dataset/dataset.sqlite

When the file already exists, CURATOR validates and reuses it. To consume a
database created by another run without loading the teacher, set:

.. code-block:: yaml

   task:
     distill:
       teacher_model_path: null
       teacher_labels_path: /path/to/dataset.sqlite

The data config is switched to the SQLite reader after label preparation.
Teacher energy, force, and Hessian targets are aligned with the student's
rescaling layers before their distillation losses are evaluated.

Teacher-label lifecycle
~~~~~~~~~~~~~~~~~~~~~~~

The offline distillation config is defined in
``curator/configs/task/distill/offline.yaml``.

.. list-table:: Offline label controls
   :header-rows: 1
   :widths: 26 74

   * - Option
     - Behavior
   * - ``teacher_model_path``
     - Checkpoint, run directory, external model spec, or list of teachers used
       to generate missing labels.
   * - ``teacher_labels_path``
     - Existing SQLite file for ``data.datapath``. When explicit train, val,
       and test paths are used, this must be a directory.
   * - ``label_scope: dataset``
     - Label the complete ``data.datapath`` once. This is the default and lets
       different random seeds or split ratios share one label store.
   * - ``label_scope: split``
     - Apply the current split first, then create ``train.sqlite``,
       ``val.sqlite``, and ``test.sqlite``.
   * - ``resume: true``
     - Compare the existing row count with the source and append only missing
       rows.
   * - ``overwrite: true``
     - Remove the existing store and regenerate all teacher labels.
   * - ``teacher_cfg``
     - Optional compatibility config used while loading an older teacher.

Before training, reused databases are checked for every required teacher
column and for non-finite floating-point values. Distillation losses use
``only_train: true`` by default, so validation reports the regular supervised
objectives.

.. note::

   ``label_scope: split`` from a single ``data.datapath`` does not currently
   support a multi-domain datamodule. Use explicit split databases or split
   paths for that case.

Hessian distillation
~~~~~~~~~~~~~~~~~~~~

Energy and force supervision constrains a potential and its first derivative.
Hessian distillation transfers local curvature information as an additional
training signal.

For a structure with ``N`` atoms, the Cartesian Hessian contains
``(3N) x (3N)`` entries. The available strategies expose different storage and
sampling trade-offs.

.. list-table:: Hessian distillation presets
   :header-rows: 1
   :widths: 33 35 32

   * - Output preset
     - Teacher data
     - Trade-off
   * - ``energy_force_hessian_distill``
     - Store the full teacher Hessian; sample entries for the student loss.
     - Direct element-wise curvature supervision with quadratic label storage.
   * - ``energy_force_projected_hessian_distill``
     - Store probe vectors and the corresponding Hessian-vector products.
     - Storage scales as ``O(3Nk)`` for ``k`` probes instead of
       ``O((3N)^2)``. This is the preferred scalable option.
   * - ``energy_force_teacher_dynamic_projected_hessian_distill``
     - Store the full teacher Hessian and project it onto fresh student probes
       during training.
     - Varying stochastic directions, while retaining full-Hessian storage.

The Hessian presets retain the supervised energy and force losses and add a
curvature loss. They do not automatically add teacher energy and force losses.

Projected-Hessian example
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   defaults:
     - task/outputs: energy_force_projected_hessian_distill
     - task/distill: offline

   data:
     datapath: /path/to/reference.traj

   task:
     distill:
       teacher_model_path: /path/to/teacher.ckpt
     projected_hessian:
       num_probes: 4
       normalize: true
       distribution: rademacher
     loss_weights:
       energy_hessian_projected_distill: 1.0

   run_path: runs/distill-projected-hessian

The matching output-parameter preset enables position-based force
differentiation and adds the required ``EnergyHessianOutput`` modules.

.. warning::

   Full and dynamic projected-Hessian distillation can produce very large
   SQLite stores. Prefer projected-Hessian labels for larger structures or
   datasets. Hessian label generation also requires a teacher whose output
   modules can be prepared for position derivatives.

Combining fine-tuning and distillation
--------------------------------------

The configuration groups compose directly. For example, a LoRA student can be
trained from offline teacher labels with:

.. code-block:: yaml

   defaults:
     - finetune: lora
     - task/outputs: energy_force_distill
     - task/distill: offline

   model_path:
     path: /path/to/student-initialization.ckpt
     mode: weights

   data:
     datapath: /path/to/reference.traj

   task:
     distill:
       teacher_model_path: /path/to/teacher.ckpt

   run_path: runs/lora-distill

Here ``model_path`` initializes the student. ``teacher_model_path`` is used
only to generate teacher targets and does not initialize the student.


Troubleshooting
---------------

Missing offline SQLite file
~~~~~~~~~~~~~~~~~~~~~~~~~~~

If ``teacher_model_path`` is null, ``teacher_labels_path`` must point to an
existing compatible store. Otherwise CURATOR cannot generate the missing
teacher columns.

Interrupted label generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set ``resume: true``. CURATOR compares completed rows with the source length
and appends the missing rows. Do not combine ``resume: true`` with
``overwrite: true`` unless the intention is to rebuild the store.

Changing the teacher
~~~~~~~~~~~~~~~~~~~~

Teacher labels are derived data. When the teacher checkpoint or requested
teacher outputs change, use a new ``teacher_labels_path`` or set
``overwrite: true``. Reusing labels from a different teacher silently changes
the intended experiment even if the database schema remains valid.

Out-of-memory during Hessian label generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reduce the structure or batch size, lower ``projected_hessian.num_probes``, or
switch from full Hessian labels to
``energy_force_projected_hessian_distill``.

Unexpected trainable parameters with LoRA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Remember that ``lora_freeze_base`` applies to wrapped modules. Inspect and
override ``task.optimizer_groups`` or narrow ``wrapper.lora_target_groups`` if
unwrapped modules must also remain fixed.
