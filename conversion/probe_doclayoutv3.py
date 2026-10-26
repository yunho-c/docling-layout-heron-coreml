#!/usr/bin/env python3
"""Fixed-800 PP-DocLayoutV3 Core ML feasibility probe, using cached weights.

Run with the existing Python 3.13 and coremltools 9.0 on PYTHONPATH.
Artifacts default to the repository's ignored output directory.
"""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import transformers
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

MODEL = "PaddlePaddle/PP-DocLayoutV3_safetensors"
REVISION = "e4f489dc536556fc1be02b973ba06756e0d4a2ac"
OUTPUT_NAMES = ("logits", "pred_boxes", "order_logits", "out_masks")


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def attention_batch_one(self, hidden_states, attention_mask=None,
                        encoder_hidden_states=None, encoder_attention_mask=None,
                        position_embeddings=None, reference_points=None,
                        spatial_shapes=None, spatial_shapes_list=None,
                        level_start_index=None, **kwargs):
    """Same bilinear deformable attention, removing only the unit batch axis.

    Core ML cannot represent upstream's rank-six offsets. Fixed batch one and
    four-coordinate references are deliberate parts of this export contract.
    """
    if position_embeddings is not None:
        hidden_states = hidden_states + position_embeddings
    q = hidden_states.shape[1]
    value = self.value_proj(encoder_hidden_states)
    if attention_mask is not None:
        value = value.masked_fill(~attention_mask[..., None], 0.0)
    offsets = self.sampling_offsets(hidden_states).reshape(q, self.n_heads, self.n_levels, self.n_points, 2)
    weights = self.attention_weights(hidden_states).reshape(q, self.n_heads, self.n_levels * self.n_points).softmax(-1)
    ref = reference_points.reshape(q, 1, 1, 1, 4)
    locations = ref[..., :2] + offsets / self.n_points * ref[..., 2:] * 0.5
    grids = 2 * locations - 1
    values = value.split([h*w for h,w in spatial_shapes_list], dim=1)
    samples = []
    for i, (h, w) in enumerate(spatial_shapes_list):
        level_value = values[i].transpose(1, 2).reshape(self.n_heads, self.d_model // self.n_heads, h, w)
        grid = grids[:, :, i].transpose(0, 1)
        samples.append(torch.nn.functional.grid_sample(level_value, grid, mode="bilinear", padding_mode="zeros", align_corners=False))
    weighted = weights.permute(1, 0, 2).unsqueeze(1)
    output = (torch.stack(samples, dim=-2).flatten(-2) * weighted).sum(-1)
    output = output.reshape(1, self.d_model, q).transpose(1, 2).contiguous()
    return self.output_proj(output), weights.reshape(1, q, self.n_heads, self.n_levels, self.n_points)


def compact_mask_to_box(mask, dtype):
    """Exact binary-mask bounding rectangle using row/column occupancy.

    Avoid four full [1,300,40000] coordinate intermediates and the FP32-max
    sentinel. All coordinates and sentinel values fit exactly in FP16.
    """
    h, w = mask.shape[-2:]
    columns, rows = mask.any(dim=-2), mask.any(dim=-1)
    x = torch.arange(w, dtype=dtype, device=mask.device)
    y = torch.arange(h, dtype=dtype, device=mask.device)
    xmin = torch.where(columns, x, float(w)).amin(-1)
    ymin = torch.where(rows, y, float(h)).amin(-1)
    xmax = (columns.to(dtype) * x).amax(-1) + 1
    ymax = (rows.to(dtype) * y).amax(-1) + 1
    nonempty = columns.any(-1).to(dtype)
    xmin, xmax = xmin * nonempty / w, xmax * nonempty / w
    ymin, ymax = ymin * nonempty / h, ymax * nonempty / h
    return torch.stack([(xmin+xmax)/2, (ymin+ymax)/2, xmax-xmin, ymax-ymin], -1)


def final_heads_decoder(self, inputs_embeds=None, encoder_hidden_states=None,
                        encoder_attention_mask=None, reference_points=None,
                        spatial_shapes=None, spatial_shapes_list=None,
                        level_start_index=None, order_head=None,
                        global_pointer=None, mask_query_head=None, norm=None,
                        mask_feat=None, **kwargs):
    """Fixed inference-only decoder: retain every refinement, only final heads."""
    from transformers.models.pp_doclayout_v3.modeling_pp_doclayout_v3 import inverse_sigmoid, PPDocLayoutV3DecoderOutput
    hidden_states = inputs_embeds
    reference_points = reference_points.sigmoid()
    for idx, layer in enumerate(self.layers):
        hidden_states = layer(hidden_states,
            object_queries_position_embeddings=self.query_pos_head(reference_points),
            encoder_hidden_states=encoder_hidden_states,
            reference_points=reference_points.unsqueeze(2), spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list, level_start_index=level_start_index,
            encoder_attention_mask=encoder_attention_mask, **kwargs)
        reference_points = (self.bbox_embed(hidden_states) + inverse_sigmoid(reference_points)).sigmoid().detach()
    out_query = norm(hidden_states)
    mask_embed = mask_query_head(out_query)
    masks = torch.bmm(mask_embed, mask_feat.flatten(start_dim=2)).reshape(1, self.num_queries, mask_feat.shape[-2], mask_feat.shape[-1])
    order = global_pointer(order_head[idx](out_query[:, -self.num_queries:]))
    return PPDocLayoutV3DecoderOutput(last_hidden_state=hidden_states,
        intermediate_hidden_states=hidden_states.unsqueeze(1),
        intermediate_reference_points=reference_points.unsqueeze(1),
        intermediate_logits=self.class_embed(out_query).unsqueeze(1),
        decoder_out_order_logits=order.unsqueeze(1), decoder_out_masks=masks.unsqueeze(1))


def patch_scalar_cast():
    # NumPy 2 rejects int(array([n])); traced aten::Int still means a scalar.
    from coremltools.converters.mil.frontend.torch import ops
    from coremltools.converters.mil import Builder as mb
    original = ops._cast

    def scalar_cast(context, node, dtype, dtype_name):
        x = ops._get_inputs(context, node, expected=1)[0]
        if x.val is not None and getattr(x.val, "shape", ()) == (1,):
            context.add(mb.const(val=dtype(x.val.reshape(())), name=node.name))
        else:
            original(context, node, dtype, dtype_name)

    ops._cast = scalar_cast


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        result = self.model(pixel_values=pixel_values)
        return tuple(getattr(result, name) for name in OUTPUT_NAMES)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=Path("output/ppdoclayoutv3-prototype"))
    ap.add_argument("--dataset", type=Path, default=Path("/Users/yunhocho/GitHub/IBEM-v1-math-detection-sample"))
    ap.add_argument("--precision", choices=["fp16", "fp32", "mixed"], default="fp16")
    ap.add_argument("--reuse-trace", action="store_true")
    ap.add_argument("--optimized", action="store_true", help="Prune unused intermediate heads and compact mask geometry")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(6)
    torch.manual_seed(0)
    patch_scalar_cast()
    prefix = "optimized" if args.optimized else "full"
    trace_path = args.output / f"{prefix}.pt"
    if args.reuse_trace:
        traced = torch.jit.load(str(trace_path)).eval()
    else:
        processor = AutoImageProcessor.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
        model = AutoModelForObjectDetection.from_pretrained(MODEL, revision=REVISION, local_files_only=True, attn_implementation="eager").eval()
        records = json.loads((args.dataset / "annotations.json").read_text())
        image_path = args.dataset / records[0]["image"]
        pixel_values = processor(images=Image.open(image_path).convert("RGB"), return_tensors="pt")["pixel_values"]
        wrapper = Wrapper(model).eval()
        with torch.inference_mode():
            reference = wrapper(pixel_values)
            np.savez(args.output / "reference.npz", pixel_values=pixel_values.numpy(), **{n: v.numpy() for n, v in zip(OUTPUT_NAMES, reference)})
            # Hoist fixed-shape anchors, preserving upstream values exactly and
            # removing trace-time in-place coordinate assignments.
            inner = model.model
            anchors, valid = inner.generate_anchors(((100, 100), (50, 50), (25, 25)))
            inner.config.anchor_image_size = 800
            inner.register_buffer("anchors", anchors)
            inner.register_buffer("valid_mask", valid)
            import types
            from transformers.models.pp_doclayout_v3.modeling_pp_doclayout_v3 import PPDocLayoutV3MultiscaleDeformableAttention
            for module in model.modules():
                if isinstance(module, PPDocLayoutV3MultiscaleDeformableAttention):
                    module.forward = types.MethodType(attention_batch_one, module)
            if args.optimized:
                import transformers.models.pp_doclayout_v3.modeling_pp_doclayout_v3 as v3
                v3.mask_to_box_coordinate = compact_mask_to_box
                inner.decoder.forward = types.MethodType(final_heads_decoder, inner.decoder)
            patched = wrapper(pixel_values)
            errors = {n: float((a-b).abs().max()) for n, a, b in zip(OUTPUT_NAMES, reference, patched)}
            print("rewrite parity", errors, flush=True)
            assert all(v == 0 for v in errors.values())
            start = time.perf_counter()
            traced = torch.jit.trace(wrapper, pixel_values, strict=False, check_trace=False)
            traced = torch.jit.freeze(traced)
            actual = traced(pixel_values)
            trace_errors = {n: float((a-b).abs().max()) for n, a, b in zip(OUTPUT_NAMES, reference, actual)}
            print("trace parity", trace_errors, flush=True)
            traced.save(str(trace_path))
        dump(args.output / f"{prefix}-conversion_provenance.json", {
            "model": MODEL, "revision": REVISION,
            "parameters": sum(p.numel() for p in model.parameters()),
            "torch": torch.__version__, "transformers": transformers.__version__,
            "coremltools": ct.__version__, "python": platform.python_version(), "macos": platform.mac_ver()[0],
            "model_source": str(__import__(model.__module__, fromlist=["x"]).__file__),
            "image": str(image_path), "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            "processor": processor.to_dict(), "config": model.config.to_dict(),
            "output_shapes": {n: list(v.shape) for n,v in zip(OUTPUT_NAMES, reference)},
            "rewrite_errors": errors, "trace_errors": trace_errors,
            "trace_seconds": time.perf_counter()-start,
            "notes": ["Fixed batch 1, 800x800; full masks and order outputs.", "Upstream eager attention; native Core ML resample for grid_sample."]})
    print("Converting", args.precision, flush=True)
    start = time.perf_counter()
    precision = ct.precision.FLOAT16 if args.precision == "fp16" else ct.precision.FLOAT32
    if args.precision == "mixed":
        def use_fp16(op):
            # Preserve small coordinate arithmetic, iterative logit/sigmoid
            # refinement, and the global-pointer score matrix in FP32.
            return not any(tuple(v.shape) in {(1,300,4), (1,300,300)} for v in op.outputs)
        precision = ct.transform.FP16ComputePrecision(op_selector=use_fp16)
    mlmodel = ct.convert(traced, convert_to="mlprogram",
        inputs=[ct.TensorType(name="pixel_values", shape=(1,3,800,800), dtype=np.float32)],
        outputs=[ct.TensorType(name=n, dtype=np.float32) for n in OUTPUT_NAMES],
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=precision,
        skip_model_load=True)
    output = args.output / f"{prefix}-{args.precision}.mlpackage"
    mlmodel.save(str(output))
    dump(args.output / f"convert-{prefix}-{args.precision}.json", {"seconds": time.perf_counter()-start, "path": str(output)})
    print("Saved", output, flush=True)


if __name__ == "__main__":
    main()
