"""Lightning manual optimization for the Small+ACVR joint phase on one GPU."""

from model import ScratchTrainer
from graph_joint import JointGraph, eligible
from curator.data import properties
import pytorch_lightning as pl
import torch


class JointTask(pl.LightningModule):
    def __init__(self, model, *, cuda_graph=False):
        super().__init__()
        self.model = model
        self.use_graph = cuda_graph
        self.joint_graph = None
        self.automatic_optimization = False
        self.training_groups = make_groups(model)

    def on_fit_start(self):
        if self.use_graph:
            if (
                self.trainer.world_size != 1
                or self.device.type != "cuda"
                or self.trainer.precision != "32-true"
            ):
                raise ValueError(
                    "Graph joint training in Lightning requires single-GPU true FP32."
                )
            self.joint_graph = JointGraph(self.training_groups)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        # Eligibility uses CPU metadata. Transfer eligible batches inside JointGraph.
        if self.joint_graph is not None and eligible(batch):
            return batch
        return super().transfer_batch_to_device(batch, device, dataloader_idx)

    def configure_optimizers(self):
        return [
            self.training_groups.physics_optimizer,
            self.training_groups.solver_optimizer,
        ]

    def training_step(self, batch, batch_idx):
        groups = self.training_groups
        adam, soap = self.optimizers()
        if (
            self.joint_graph is not None
            and batch[properties.positions].device.type == "cpu"
        ):
            self.joint_graph.step(batch, "joint", optimizers=(adam, soap))
            return
        for opt in (adam, soap):
            opt.zero_grad(set_to_none=True)
        out = self.model(batch)
        physical_loss = groups.supervision(out, batch)["loss"]
        residual = out["variational_prediction"]["residuals"]["stationarity"]["values"]
        self.manual_backward(physical_loss, inputs=groups.physical, retain_graph=True)
        self.manual_backward(30 * residual.square().mean(), inputs=groups.proposal)
        for params in (groups.physical, groups.proposal):
            torch.nn.utils.clip_grad_norm_(params, 10.0, error_if_nonfinite=True)
        adam.step()
        soap.step()


def make_groups(model):
    return ScratchTrainer(
        model,
        backbone_lr=1.0e-4,
        energy_lr=1.0e-5,
        proposal_lr=1.0e-4,
        energy_weight=200.0,
        force_weight=200.0,
    )
