# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
#
# Pure PyTorch implementation — no CUDA compilation required.
# Forward uses F.grid_sample; backward is computed through PyTorch autograd.
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    """Multi-scale deformable attention core — pure PyTorch forward.

    This is the forward-only version that builds a full computation graph,
    allowing autograd to handle backward automatically.

    Args:
        value:                (N, S, M, D_)  value projections
        value_spatial_shapes: (L, 2)         [(H_0, W_0), (H_1, W_1), ...]
        sampling_locations:   (N, Lq, M, L, P, 2)
        attention_weights:    (N, Lq, M, L, P)

    Returns:
        output: (N, Lq, M * D_)
    """
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape

    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)

    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # value_l_:  [N, H_*W_, M, D] -> [N, H_*W_, M*D] -> [N, M*D, H_*W] -> [N*M, D, H_, W_]
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_ * M_, D_, H_, W_)
        # sampling_grid_l_: [N, Lq, M, P, 2] -> [N, M, Lq, P, 2] -> [N*M, Lq, P, 2]
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        # [N*M, D, Lq, P]
        sampling_value_l_ = F.grid_sample(
            value_l_, sampling_grid_l_,
            mode='bilinear', padding_mode='zeros', align_corners=False,
        )
        sampling_value_list.append(sampling_value_l_)

    # attn weights: [N, Lq, M, L, P] -> [N, M, Lq, L, P] -> [N*M, 1, Lq, L*P]
    attention_weights = attention_weights.transpose(1, 2).reshape(N_ * M_, 1, Lq_, L_ * P_)
    # stacked: [N*M, D, Lq, L, P] -> flatten(-2) -> [N*M, D, Lq, L*P]
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_ * D_, Lq_)
    return output.transpose(1, 2).contiguous()


class MSDeformAttnFunction(Function):
    """Pure PyTorch multi-scale deformable attention, with autograd.

    Drop-in replacement for the CUDA-based MSDeformAttnFunction.
    The forward pass uses ``ms_deform_attn_core_pytorch`` (built on
    ``F.grid_sample``), and the backward pass re-runs the forward graph
    with ``torch.autograd.grad`` so that all gradient flow is handled
    automatically.
    """

    @staticmethod
    def forward(ctx, value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights, im2col_step):
        # ── save for backward ──────────────────────────────────────────
        ctx.save_for_backward(value, sampling_locations, attention_weights)
        ctx.value_spatial_shapes = value_spatial_shapes

        # ── forward ────────────────────────────────────────────────────
        output = ms_deform_attn_core_pytorch(
            value, value_spatial_shapes, sampling_locations, attention_weights,
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        value, sampling_locations, attention_weights = ctx.saved_tensors
        spatial_shapes = ctx.value_spatial_shapes

        # Re-create the computation graph so autograd can trace backward
        value = value.detach().requires_grad_(True)
        sampling_locations = sampling_locations.detach().requires_grad_(True)
        attention_weights = attention_weights.detach().requires_grad_(True)

        with torch.enable_grad():
            output = ms_deform_attn_core_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights,
            )

        grad_value, grad_sampling_loc, grad_attn_weight = torch.autograd.grad(
            output, [value, sampling_locations, attention_weights], grad_output,
        )

        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


def ms_deform_attn_pytorch(value, value_spatial_shapes, value_level_start_index,
                           sampling_locations, attention_weights, im2col_step=64):
    """Convenience wrapper around ``MSDeformAttnFunction.apply``.

    Has the same signature as the original CUDA ``ms_deform_attn_forward``,
    making it a drop-in replacement everywhere ``MSDeformAttnFunction.apply``
    is called.
    """
    return MSDeformAttnFunction.apply(
        value, value_spatial_shapes, value_level_start_index,
        sampling_locations, attention_weights, im2col_step,
    )
