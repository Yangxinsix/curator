# Curator Backend Runtimes

Foundation MLIP packages are routed per backend runtime. Curator core does not
import every backend package directly, but heavy base packages such as
Torch/CUDA should come from the host or a shared base Python environment.

Local setup is host-first. If the host Python already provides a usable backend
module, Curator MCP uses the host Python and does not create or sync a runtime
venv. Some backends have stricter checks than plain importability; for example,
MatGL must be new enough to load the current pretrained registry:

```bash
python -c "import torch, mace, matgl"
sync_backend_runtime("mace")   # no-op when mace is already importable
sync_backend_runtime("matgl")  # no-op only when the installed MatGL is usable
```

For a missing backend, `sync_backend_runtime` creates `runtimes/<backend>/.venv`
with `--system-site-packages`, uses the current/base Python interpreter, and
installs only backend-specific packages while constraining Torch-family base
packages to the versions already installed in the base Python. The base Python
must already provide `torch`:

```bash
sync_backend_runtime("orb")
sync_backend_runtime("nequip", base_python="/path/to/python-with-torch")
```

Use `mode="isolated"` only as an explicit fallback when a separate Torch/CUDA
stack is truly required. Isolated mode will run `uv sync --project` and may
download large Torch/CUDA wheels.

Curator MCP calls the selected backend runner with either host Python or the
runtime venv Python:

```bash
python -m curator_mcp.backend_runner \
  --request runs/job_001/request.json \
  --out runs/job_001
```

The local backend runner currently supports:

- `health`: check whether the runtime package imports are available.
- `resolve` / `fetch` / `probe`: resolve registry aliases and optionally load the model.
- `predict`: read an ASE-compatible `atoms_path`, load the resolved adapter spec, and write normalized energy/forces/stress output to `predictions.json`.

Prediction/probe calls never install dependencies as a side effect. Runtime
dependency installation must be explicit.
