import torch
from e3nn.util.jit import compile_mode
from typing import Optional, Any, Union

class Interaction(torch.nn.Module):
    """Abstract class for message layer in message-passing neural networks"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def exchange_info(
        self, 
        node_feat, 
        lammps_data: Optional[Any], 
        n_ghost: Optional[int],
        is_first_layer: bool = False,
    ):
        # if not using MLIAP package in Lammps, then disable message-passing
        if lammps_data is None or is_first_layer or torch.jit.is_scripting():
            return node_feat
        
        pad = torch.zeros(
            (n_ghost, node_feats.shape[1]),
            dtype=node_feats.dtype,
            device=node_feats.device,
        )
        node_feats = torch.cat((node_feats, pad), dim=0)
        node_feats = LammpsMessagePassing.apply(node_feats, lammps_data)
        return node_feats
    
    def truncate_ghost(
        self,
        tensor: torch.Tensor,
        nlocal: Optional[int],
    ):
        # truncate features or grads that belong to ghost atoms
        return tensor[:nlocal] if nlocal is not None else tensor

class LammpsMessagePassing(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        node_feat, data = args  # unpack
        ctx.vec_len = node_feat.shape[-1]
        ctx.data = data
        out = torch.empty_like(node_feat)
        data.forward_exchange(node_feat, out, node_feat)
        return out

    @staticmethod
    def backward(ctx, *grad_outputs):
        (grad,) = grad_outputs  # unpack
        gout = torch.empty_like(grad)
        ctx.data.reverse_exchange(grad, gout, ctx.vec_len)
        return gout, None