import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from curator.model import LitNNP
from torch_ema import ExponentialMovingAverage as EMA

class ExponentialMovingAverage(Callback):
    def __init__(self, decay: float, use_num_updates: bool=True, *args, **kwargs):
        self.decay = decay
        self.ema = None
        self.use_num_updates = use_num_updates
        self._to_load = None

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: LitNNP):
        if self.ema is None:
            self.ema = EMA(pl_module.model.parameters(), decay=self.decay, use_num_updates=self.use_num_updates)
        if self._to_load is not None:
            self.ema.load_state_dict(self._to_load)
            self._to_load = None

        # load average parameters, to have same starting point as after validation
        self.ema.store()
        self.ema.copy_to()

    def on_train_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self.ema.restore()

    def on_train_batch_end(self, trainer, pl_module: LitNNP, *args, **kwargs):
        self.ema.update()

    def on_validation_epoch_start(
        self, trainer: "pl.Trainer", pl_module: LitNNP, *args, **kwargs
    ):
        self.ema.store()
        self.ema.copy_to()

    def load_state_dict(self, state_dict):
        if "ema" in state_dict:
            if self.ema is None:
                self._to_load = state_dict["ema"]
            else:
                self.ema.load_state_dict(state_dict["ema"])

    def state_dict(self):
        return {"ema": self.ema.state_dict() if self.ema is not None else None}