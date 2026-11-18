"""CUDA Graph replay of a fixed-Q molecular joint step. Optimizers stay eager."""

from collections import OrderedDict
import torch
from curator.data import properties as p
from curator.train.train import tree_to_device
from fixed_q_batch import panel_energy_charge


def eligible(batch, *, periodic=False):
    """Exact subset for the batched E/F and fixed-total-charge loss formulas."""
    v = batch["variational"]
    n = batch[p.n_atoms].reshape(-1)
    count = len(n)
    if (
        not count
        or batch[p.positions].device.type != "cpu"
        or not bool((batch[p.pbc] == periodic).all())
        or not bool((n == n[0]).all())
        or int(n[0]) < 1
        or p.total_charge not in batch
        or v["condition_inputs"]
        or len(v["boundary_spec"]) != count
        or not torch.equal(v["condition_geometry_index"], torch.arange(count))
        or not torch.equal(
            batch[p.image_idx], torch.arange(count).repeat_interleave(int(n[0]))
        )
        or not torch.equal(v["condition_state_ptr"], torch.arange(count + 1) * n[0])
        or not bool((v["condition_weight"] == 1).all())
        or not bool((batch["weight"] == 1).all())
    ):
        return False
    totals = batch[p.total_charge].reshape(-1)
    for i, spec in enumerate(v["boundary_spec"]):
        groups, eq = spec["groups"], spec["equalities"]
        if (
            spec["controls"]
            or spec.get("environment")
            or spec.get("partition_groups")
            or spec.get("gauge")
            or spec["variable"]["kind"] != "atomic_charge"
            or spec["variable"]["components"] != 1
            or spec["variable"]["unit"] != "e"
            or spec["variable"]["conjugate_unit"] != "V"
            or len(groups) != 1
            or groups[0]["selection"] != "all"
            or len(eq) != 1
            or eq[0]["kind"] != "group_sum"
            or eq[0]["unit"] != "e"
            or eq[0]["components"]
            or len(eq[0]["terms"]) != 1
            or eq[0]["target"] != float(totals[i])
        ):
            return False
        term = eq[0]["terms"][0]
        if (
            term["group"] != groups[0]["name"]
            or term["coefficient"] != 1
            or term["weight_field"]
            or term["weights"]
        ):
            return False
    atoms = int(n[0])
    for name, size, shape, unit in (
        ("condition_energy", 1, (1,), "eV"),
        ("condition_forces", 3 * atoms, (atoms, 3), "eV/Angstrom"),
    ):
        label = v["labels"].get(name)
        if (
            not label
            or label["unit"] != unit
            or any(
                meta.get(
                    "energy_kind" if name == "condition_energy" else "force_potential"
                )
                != ("internal" if name == "condition_energy" else "thermodynamic")
                for meta in label["metadata"]
            )
            or not bool((label["present"] & label["eligible"]).all())
            or tuple(label["shape"]) != (shape,) * count
            or not torch.equal(label["ptr"], torch.arange(count + 1) * size)
        ):
            return False
    return True


class JointGraph:
    eligible = staticmethod(eligible)

    def __init__(self, trainer, *, max_shapes=2, pure=False):
        self.trainer = trainer
        self.pure = pure
        self.cache = OrderedDict()
        self.max_shapes = max_shapes
        self.stream = torch.cuda.Stream(device=trainer.physical[0].device)
        self.captures = 0
        self.replays = 0
        self.fallbacks = 0

    def _inputs(self, batch):
        model = self.trainer.model
        # Rebuild the real neighbors on every step. Pad only the edge list,
        # using an out-of-cutoff constant vector with identically zero derivatives.
        data = tree_to_device(batch, self.trainer.physical[0].device)
        with torch.no_grad():
            for layer in model.input_modules:
                data = layer(data)
            model.representation.embeddings["onehot"](data)
        edges = data[p.edge_idx]
        length = (len(edges) + 255) // 256 * 256 or 256
        edge_idx = torch.nn.functional.pad(edges, (0, 0, 0, length - len(edges)))
        mask = (torch.arange(length, device=edges.device) < len(edges))[:, None]
        labels = data["variational"]["labels"]
        inputs = (
            data[p.positions].detach(),
            data[p.atomic_types],
            data[p.total_charge],
            edge_idx,
            mask,
            labels["condition_energy"]["values"],
            labels["condition_forces"]["values"],
        )
        return data, inputs

    @staticmethod
    def _edge_vectors(pos, edges, mask, outside):
        return torch.where(mask, pos[edges[:, 1]] - pos[edges[:, 0]], outside)

    def _capture(self, data, inputs, physical):
        t = self.trainer
        model = t.model
        static = tuple(v.detach().clone() for v in inputs)
        static[0].requires_grad_(True)
        pos, atom_types, total, edges, mask, y_e, y_f, *extra = static
        atoms = pos.shape[0] // total.numel()
        # During capture every leaf participating in backward must have been
        # allocated or had its AccumulateGrad created on the capture stream.
        outside = pos.new_tensor([2 * model.representation.cutoff, 0.0, 0.0])

        def differentiate():
            d = dict(data)
            d[p.positions] = pos
            d[p.atomic_types] = atom_types
            d[p.total_charge] = total
            d[p.edge_idx] = edges
            d[p.edge_diff] = self._edge_vectors(pos, edges, mask, outside, *extra)
            d = model.representation(d)
            if self.pure:
                energy = d[p.energy].reshape(-1)
                dx = torch.autograd.grad(energy.sum(), pos, create_graph=True)[0]
            else:
                energy, charge, dq = panel_energy_charge(
                    model.output_modules[0], d, return_partial=True
                )
                if dq is None:
                    dx, dq = torch.autograd.grad(
                        energy.sum(), (pos, charge), create_graph=True
                    )
                else:
                    dx = torch.autograd.grad(energy.sum(), pos, create_graph=True)[0]
                residual = dq - dq.mean(1, keepdim=True)
            ew, fw = [
                t.supervision.supervised_weights[k]
                / t.supervision.supervised_scales[k] ** 2
                for k in ("condition_energy", "condition_forces")
            ]
            phys = (
                ew * ((energy - y_e) / atoms).square().mean()
                + fw * (dx.flatten() + y_f).square().mean()
            )
            if self.pure:
                gp = torch.autograd.grad(phys, t.backbone, allow_unused=True)
                return phys, phys.new_zeros(()), gp, ()
            mse = residual.square().mean()
            gp = torch.autograd.grad(
                phys, physical, retain_graph=True, allow_unused=True
            )
            gs = torch.autograd.grad(30 * mse, t.proposal, allow_unused=True)
            return phys, mse, gp, gs

        # Give the graph its own input leaves. Parameter AccumulateGrad handles
        # from eager training do not need to be used: only autograd.grad is called.
        for _ in range(2):
            differentiate()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=self.stream):
            result = differentiate()
        self.captures += 1
        return graph, static, result, (outside, data)

    def step(self, batch, phase, *, optimizers=None, freeze_backbone=False):
        t = self.trainer
        if self.pure and freeze_backbone:
            raise ValueError("A pure-backbone fit needs trainable backbone parameters")
        physical = t.energy if freeze_backbone else t.physical
        if (
            phase != "joint"
            or t.physical[0].device.type != "cuda"
            or not self.eligible(batch)
        ):
            self.fallbacks += 1
            if self.pure:
                raise ValueError(
                    "Pure graph benchmark requires an eligible joint batch"
                )
            if freeze_backbone:
                full = t.physical
                t.physical = physical
                try:
                    return t.step(batch, phase)
                finally:
                    t.physical = full
            return t.step(batch, phase)
        t.model.train()
        current = torch.cuda.current_stream(t.physical[0].device)
        self.stream.wait_stream(current)
        with torch.cuda.stream(self.stream):
            data, inputs = self._inputs(batch)
            key = (freeze_backbone, tuple((i.shape, i.dtype) for i in inputs))
            if key not in self.cache:
                if len(self.cache) == self.max_shapes:
                    self.cache.popitem(last=False)
                self.cache[key] = self._capture(data, inputs, physical)
            g, static, result, _keepalive = self.cache[key]
            self.cache.move_to_end(key)
            with torch.no_grad():
                for dest, source in zip(static, inputs):
                    dest.copy_(source)
            g.replay()
            self.replays += 1
            phys, mse, gp, gs = result
            if not bool(torch.isfinite(torch.stack((phys, mse))).all()):
                raise FloatingPointError("Nonfinite CUDA graph loss")
            reports = {
                "physical_loss": float(phys.detach()),
                "stationarity_rms_V": float(mse.detach().sqrt()),
            }
            adam, soap = (
                (t.physics_optimizer, t.solver_optimizer)
                if optimizers is None
                else optimizers
            )
            groups = (
                ("physical", physical, gp, adam),
                ("proposal", t.proposal, gs, soap),
            )
            if self.pure:
                groups = (("physical", t.backbone, gp, t.physics_optimizer),)
            for name, params, grads, opt in groups:
                opt.zero_grad(set_to_none=True)
                for param, grad in zip(params, grads):
                    param.grad = grad
                reports[name + "_grad_norm"] = float(
                    torch.nn.utils.clip_grad_norm_(
                        params, 10.0, error_if_nonfinite=True
                    )
                )
            for _, _, _, opt in groups:
                opt.step()
        current.wait_stream(self.stream)
        return reports
