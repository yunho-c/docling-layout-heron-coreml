#!/usr/bin/env python
"""Convert docling-layout-heron (RT-DETRv2) to a native Core ML .mlpackage.

Input contract matches docling-rs `crates/docling-pdf/src/layout.rs`:
page image resized to 640x640, scaled to [0,1], NO mean/std normalization
(the heron preprocessor_config has do_normalize=false).

Outputs: (logits, pred_boxes) raw -- no HF post-processing, no dict.
"""
import sys, torch, torch.nn as nn, coremltools as ct
from transformers import RTDetrV2ForObjectDetection

REPO = "docling-project/docling-layout-heron"
OUT = "layout_heron.mlpackage"


# --- op fix #1 ---------------------------------------------------------------
# RTDetrV2's decoder calls `torch.nn.functional.grid_sample` in its multi-scale
# deformable attention (`multi_scale_deformable_attention_v2`). coremltools 8.3
# has no torch->MIL lowering for `grid_sampler`, so the trace fails with
# "convert function for op 'grid_sampler' not implemented". We register a shim
# that expresses bilinear grid_sample with align_corners=False, padding=zeros
# using only gather/arithmetic ops -- all of which lower cleanly.
from coremltools.converters.mil.frontend.torch.torch_op_registry import (
    register_torch_op, _TORCH_OPS_REGISTRY)
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.frontend.torch.ops import _get_inputs


def _grid_sample_bilinear_zeros(x, grid):
    """Pure-torch bilinear grid_sample (align_corners=False, zero padding).

    Used as a traced replacement for F.grid_sample so the exported graph
    contains only gather/mul/add. x: (N,C,H,W), grid: (N,Hg,Wg,2) in [-1,1].
    """
    N, C, H, W = x.shape
    _, Hg, Wg, _ = grid.shape
    gx = grid[..., 0]
    gy = grid[..., 1]
    # align_corners=False mapping
    ix = ((gx + 1) * W - 1) / 2
    iy = ((gy + 1) * H - 1) / 2
    ix0 = torch.floor(ix)
    iy0 = torch.floor(iy)
    ix1 = ix0 + 1
    iy1 = iy0 + 1
    wa = (ix1 - ix) * (iy1 - iy)
    wb = (ix1 - ix) * (iy - iy0)
    wc = (ix - ix0) * (iy1 - iy)
    wd = (ix - ix0) * (iy - iy0)

    def valid(xx, yy):
        return ((xx >= 0) & (xx <= W - 1) & (yy >= 0) & (yy <= H - 1)).to(x.dtype)

    va, vb = valid(ix0, iy0), valid(ix0, iy1)
    vc, vd = valid(ix1, iy0), valid(ix1, iy1)
    ix0c = ix0.clamp(0, W - 1).long(); ix1c = ix1.clamp(0, W - 1).long()
    iy0c = iy0.clamp(0, H - 1).long(); iy1c = iy1.clamp(0, H - 1).long()
    flat = x.reshape(N, C, H * W)

    def take(xx, yy):
        idx = (yy * W + xx).reshape(N, 1, Hg * Wg).expand(N, C, Hg * Wg)
        return torch.gather(flat, 2, idx).reshape(N, C, Hg, Wg)

    out = (take(ix0c, iy0c) * (wa * va).unsqueeze(1)
           + take(ix0c, iy1c) * (wb * vb).unsqueeze(1)
           + take(ix1c, iy0c) * (wc * vc).unsqueeze(1)
           + take(ix1c, iy1c) * (wd * vd).unsqueeze(1))
    return out


# --- op fix #2 ---------------------------------------------------------------
# RTDetrV2's hybrid encoder does `int(width)` / `int(height)` when it builds the
# 2D sin-cos position embedding; under tracing `width` is a 1-element *tensor*,
# so the traced graph carries an `aten::Int` whose input is a const of shape
# (1,). coremltools' `_cast` does `dtype(x.val)` and numpy 2 refuses to convert
# a 1-d array to a Python scalar. Squeeze size-1 arrays before the cast --
# semantically identical, and it is exactly what `aten::Int` means.
from coremltools.converters.mil.frontend.torch import ops as _ct_ops
_ct_orig_cast = _ct_ops._cast


def _cast_squeeze1(context, node, dtype, dtype_str):
    inputs = _ct_ops._get_inputs(context, node, expected=1)
    x = inputs[0]
    if x.val is not None and getattr(x.val, "shape", ()) == (1,):
        from coremltools.converters.mil import Builder as _mb
        res = _mb.const(val=dtype(x.val.reshape(())), name=node.name)
        context.add(res)
        return
    return _ct_orig_cast(context, node, dtype, dtype_str)


_ct_ops._cast = _cast_squeeze1
for _k in ("int", "float", "bool"):
    if _k in _TORCH_OPS_REGISTRY.__dict__.get("_all_ops", {}):
        pass
# the aten::Int/Float/Bool handlers close over the module-global `_cast`, so
# rebinding the module attribute above is enough for coremltools 8.3.



# --- op fix #3 ---------------------------------------------------------------
# `RTDetrV2Model.generate_anchors` divides through an ellipsis slice in place:
#     grid_xy[..., 0] /= width ; grid_xy[..., 1] /= height
# The trace records `_internal_op_tensor_inplace_copy_`, and coremltools 8.3
# lowers it to `slice_update` with a mis-broadcast update shape -> "The update
# tensor should have shape [80,80,2]. Got (1,80,80)". Replace the method with a
# functionally identical, out-of-place version (divide by a [width, height]
# vector), which lowers to a plain `mul`.
def _generate_anchors(self, spatial_shapes=None, grid_size=0.05,
                      device="cpu", dtype=torch.float32):
    if spatial_shapes is None:
        spatial_shapes = [
            [int(self.config.anchor_image_size[0] / s),
             int(self.config.anchor_image_size[1] / s)]
            for s in self.config.feat_strides
        ]
    anchors = []
    for level, (height, width) in enumerate(spatial_shapes):
        height = int(height)
        width = int(width)
        grid_y, grid_x = torch.meshgrid(
            torch.arange(end=height, device=device).to(dtype),
            torch.arange(end=width, device=device).to(dtype),
            indexing="ij",
        )
        grid_xy = torch.stack([grid_x, grid_y], -1).unsqueeze(0) + 0.5
        denom = torch.tensor([width, height], dtype=dtype, device=device)
        grid_xy = grid_xy / denom
        wh = torch.ones_like(grid_xy) * grid_size * (2.0 ** level)
        anchors.append(torch.concat([grid_xy, wh], -1).reshape(-1, height * width, 4))
    eps = 1e-2
    anchors = torch.concat(anchors, 1)
    valid_mask = ((anchors > eps) * (anchors < 1 - eps)).all(-1, keepdim=True)
    anchors = torch.log(anchors / (1 - anchors))
    anchors = torch.where(
        valid_mask, anchors,
        torch.tensor(torch.finfo(dtype).max, dtype=dtype, device=device))
    return anchors, valid_mask



class Wrapper(nn.Module):
    """forward(pixel_values: (1,3,640,640) f32 in [0,1]) -> (logits, pred_boxes)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        out = self.model(pixel_values=pixel_values, return_dict=True)
        return out.logits, out.pred_boxes


def main():
    import transformers.models.rt_detr_v2.modeling_rt_detr_v2 as m
    # op fix #1 applied at the source: swap F.grid_sample for the traceable
    # decomposition inside the deformable-attention helper.
    orig = torch.nn.functional.grid_sample

    def patched(input, grid, mode="bilinear", padding_mode="zeros",
                align_corners=None):
        if mode == "bilinear" and padding_mode == "zeros" and not align_corners:
            return _grid_sample_bilinear_zeros(input, grid)
        return orig(input, grid, mode, padding_mode, align_corners)

    torch.nn.functional.grid_sample = patched
    m.F.grid_sample = patched
    # op fix #3
    m.RTDetrV2Model.generate_anchors = _generate_anchors

    model = RTDetrV2ForObjectDetection.from_pretrained(REPO, torch_dtype=torch.float32)
    model.eval()
    w = Wrapper(model).eval()

    ex = torch.rand(1, 3, 640, 640, dtype=torch.float32)
    with torch.no_grad():
        ref = w(ex)
    print("torch out:", [tuple(t.shape) for t in ref])

    with torch.no_grad():
        ts = torch.jit.trace(w, ex, strict=False)
    ts = torch.jit.freeze(ts)

    mlmodel = ct.convert(
        ts,
        convert_to="mlprogram",
        inputs=[ct.TensorType(name="pixel_values", shape=(1, 3, 640, 640),
                              dtype=__import__("numpy").float32)],
        outputs=[ct.TensorType(name="logits"), ct.TensorType(name="pred_boxes")],
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
    )
    mlmodel.save(OUT)
    print("saved", OUT)


if __name__ == "__main__":
    main()
