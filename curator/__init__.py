from curator import (
    data,
    layer,
    model,
    select,
    simulate,
    label,
)

import os
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import torch
# ensure delete always succeeds
torch._utils._thread_local_state.map_location = None
