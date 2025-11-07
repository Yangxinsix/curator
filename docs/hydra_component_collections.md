# Reusable Hydra component bundles

Curator's Hydra presets now split trainer callbacks, model module stacks, and task outputs into
small component files that you can mix and match. Each component lives in its own config
file and the top-level presets reference them through Hydra's package syntax. The runtime still
accepts the legacy list-based configs because the CLI normalizes dictionaries back into the
sequences that Lightning expects.

## Where the components live

| Feature | Directory | Example component |
| --- | --- | --- |
| Trainer callbacks | `curator/configs/trainer/callbacks/` | `model_checkpoint.yaml` |
| Model input modules | `curator/configs/model/input_modules/` | `pairwise_distance.yaml` |
| Model output modules | `curator/configs/model/output_modules/` | `gradient_output.yaml` |
| Task outputs | `curator/configs/task/outputs/components/` | `energy.yaml` |

The default presets include these components via their `defaults` lists, so existing
invocations keep working. For example, `curator/configs/trainer/default_trainer.yaml`
now imports individual callback files instead of embedding a single monolithic list.

## Building custom combinations

Use Hydra's package override syntax to compose just the pieces you need. A minimal trainer
that only enables early stopping and EMA can be written as:

```yaml
# my_trainer.yaml
defaults:
  - _self_
  - /trainer/callbacks@callbacks.early_stopping: early_stopping
  - /trainer/callbacks@callbacks.ema: ema
```

Run it with:

```bash
python -m curator.cli train trainer=@my_trainer
```

The same pattern works for the model and task outputs:

```yaml
# my_model.yaml
defaults:
  - _self_
  - representation: painn
  - /model/input_modules@input_modules.pairwise_distance: pairwise_distance
  - /model/output_modules@output_modules.atomwise_reduce: atomwise_reduce
  - /model/output_modules@output_modules.global_rescale_shift: global_rescale_shift
```

```yaml
# my_outputs.yaml
defaults:
  - _self_
  - /task/outputs/components@energy: energy_per_atom
  - /task/outputs/components@forces: forces_per_species
```

Then compose them at launch time:

```bash
python -m curator.cli train \
  trainer=@my_trainer \
  model=@my_model \
  task.outputs=@my_outputs
```

Because each component is keyed by name, you can still override a single parameter via the
command line, e.g. `trainer.callbacks.ema.decay=0.999` or `task.outputs.energy.loss_weight=2.0`.

## Daily workflow tips

1. Start from the bundled presets (`default_trainer`, `nnp`, and `energy_force`). They already
   import the standard components so you only need to add or override the pieces you care about.
2. To reuse a bundle, create a lightweight config in your experiment folder that lists the
   desired components. Hydra will merge everything automatically when you pass `@my_config`
   on the CLI or add it to a `defaults` list.
3. When experimenting from the shell, you can temporarily add a component without editing files
   using `+` syntax, e.g. `+trainer/callbacks@callbacks.gradient_clip=gradient_clip`. The CLI
   will convert the resulting dictionaries back into callback lists before instantiating the
   Lightning trainer, so both new and legacy configs behave the same way.

For reference, see the component-driven presets in:

- `curator/configs/trainer/default_trainer.yaml`
- `curator/configs/model/nnp.yaml`
- `curator/configs/task/outputs/energy_force.yaml`
