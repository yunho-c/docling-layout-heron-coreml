#!/usr/bin/env python3
"""Inspect placement and measure complete Python Core ML prediction calls."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

import coremltools as ct
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import expit


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2) + "\n")


def detections(output, threshold=0.3):
    scores = expit(output["logits"][0]).reshape(-1)
    indices = np.argsort(-scores, kind="stable")[:300]
    indices = indices[scores[indices] >= threshold]
    query, labels = indices // 25, indices % 25
    b = output["pred_boxes"][0, query]
    xyxy = np.concatenate([b[:, :2] - b[:, 2:]/2, b[:, :2]+b[:, 2:]/2], axis=1)
    return {"query": query, "labels": labels, "scores": scores[indices], "boxes": xyxy}


def compare(reference, actual):
    stats = {}
    for name in actual:
        a, b = reference[name], actual[name]
        stats[name] = {"shape": list(b.shape), "finite": bool(np.isfinite(b).all()),
                       "max_abs": float(np.max(np.abs(a-b))), "mean_abs": float(np.mean(np.abs(a-b)))}
    if any(not np.isfinite(value).all() for name, value in actual.items() if name != "order_logits") or np.isnan(actual["order_logits"]).any():
        return {"raw": stats}
    r, a = detections(reference), detections(actual)
    rb, ab = r["boxes"], a["boxes"]
    lo = np.maximum(rb[:,None,:2], ab[None,:,:2])
    hi = np.minimum(rb[:,None,2:], ab[None,:,2:])
    intersection = np.prod(np.maximum(hi-lo, 0), axis=-1)
    union = np.prod(rb[:,2:]-rb[:,:2], axis=-1)[:,None] + np.prod(ab[:,2:]-ab[:,:2],axis=-1)[None,:]-intersection
    iou = intersection / np.maximum(union, 1e-10)
    candidate = (r["labels"][:,None] == a["labels"][None,:]) & (iou >= .5)
    # One additional valid pair outweighs every possible IoU tie-break gain.
    ri, ai = linear_sum_assignment(np.where(candidate, min(len(rb), len(ab))+1+iou, 0), maximize=True)
    valid = candidate[ri, ai]
    ri, ai = ri[valid], ai[valid]
    rq, aq = r["query"][ri], a["query"][ai]
    masks = []
    if "out_masks" in actual:
        for x,y in zip(rq, aq):
            rm, am = reference["out_masks"][0,x] > 0, actual["out_masks"][0,y] > 0
            masks.append(float(np.count_nonzero(rm & am) / max(np.count_nonzero(rm | am), 1)))
    # Upstream _get_order_seqs: sigmoid upper-triangle pairwise logits, add
    # reverse preferences, sum columns, argsort; use the processor itself.
    import torch
    from transformers import AutoImageProcessor
    processor = AutoImageProcessor.from_pretrained("PaddlePaddle/PP-DocLayoutV3_safetensors", revision="e4f489dc536556fc1be02b973ba06756e0d4a2ac", local_files_only=True)
    ranks_r = processor._get_order_seqs(torch.from_numpy(np.array(reference["order_logits"], copy=True)))[0].numpy()[rq]
    ranks_a = processor._get_order_seqs(torch.from_numpy(np.array(actual["order_logits"], copy=True)))[0].numpy()[aq]
    pairs = np.triu_indices(len(ri), 1)
    order_matches = (ranks_r[:,None] < ranks_r[None,:])[pairs] == (ranks_a[:,None] < ranks_a[None,:])[pairs]
    return {"raw": stats, "detection": {
        "threshold": .3, "reference_count": len(rb), "actual_count": len(ab), "matched": len(ri),
        "missing": len(rb)-len(ri), "extra": len(ab)-len(ai),
        "mean_iou": float(iou[ri,ai].mean()) if len(ri) else None,
        "min_iou": float(iou[ri,ai].min()) if len(ri) else None,
        "score_abs_mean": float(np.abs(r["scores"][ri]-a["scores"][ai]).mean()) if len(ri) else None,
        "score_abs_max": float(np.abs(r["scores"][ri]-a["scores"][ai]).max()) if len(ri) else None,
        "mask_binary_iou_mean": float(np.mean(masks)) if masks else None,
        "order_pair_count": len(order_matches), "order_pair_agree": int(order_matches.sum()),
        "matches": [[int(x),int(y),float(z)] for x,y,z in zip(rq, aq, iou[ri,ai])]},
        "predictions": {k: v.tolist() for k,v in a.items()}}


def device_name(device):
    return type(device).__name__.replace("ML", "").replace("ComputeDevice", "")


def save_plan(model, units, path):
    plan = ct.models.compute_plan.MLComputePlan.load_from_path(model.get_compiled_model_path(), compute_units=units)
    ops = []
    def visit(block):
        for op in block.operations:
            if op.operator_name != "const":
                usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
                cost = plan.get_estimated_cost_for_mlprogram_operation(op)
                ops.append({"operator": op.operator_name, "outputs": [x.name for x in op.outputs],
                    "preferred": device_name(usage.preferred_compute_device) if usage else "unknown",
                    "supported": [device_name(x) for x in usage.supported_compute_devices] if usage else [],
                    "cost": cost.weight if cost else None})
            for child in op.blocks:
                visit(child)
    for function in plan.model_structure.program.functions.values():
        visit(function.block)
    counts = dict(Counter(op["preferred"] for op in ops))
    costs = defaultdict(float)
    for op in ops:
        costs[op["preferred"]] += op["cost"] or 0
    summary = {"counts": counts, "estimated_cost_weights": dict(costs)}
    dump(path, {**summary, "operations": ops, "note": "Compiler plan, not runtime utilization or measured device time."})
    print("Plan", summary, flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", type=Path)
    ap.add_argument("--units", choices=["cpu-ne", "all", "cpu"], default="cpu-ne")
    ap.add_argument("--reference", type=Path, default=Path("output/ppdoclayoutv3-prototype/reference.npz"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--skip-plan", action="store_true", help="Reuse placement evidence; benchmark in a fresh process")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reference = dict(np.load(args.reference))
    units = {"cpu-ne": ct.ComputeUnit.CPU_AND_NE, "all": ct.ComputeUnit.ALL, "cpu": ct.ComputeUnit.CPU_ONLY}[args.units]
    start = time.perf_counter()
    model = ct.models.MLModel(str(args.model), compute_units=units)
    load_ms = (time.perf_counter()-start)*1000
    print("load_ms", load_ms, flush=True)
    plan = None if args.skip_plan else save_plan(model, units, args.output / "compute_plan.json")
    inputs = {"pixel_values": reference.pop("pixel_values")}
    warmup, timings = [], []
    for i in range(3+args.iterations):
        start = time.perf_counter()
        output = model.predict(inputs)
        ms = (time.perf_counter()-start)*1000
        (warmup if i < 3 else timings).append(ms)
        print(i, round(ms,3), flush=True)
    np.savez(args.output / "output.npz", **output)
    parity = compare(reference, output)
    summary = {"model": str(args.model), "units": args.units, "load_ms": load_ms,
        "plan": plan, "warmup_ms": warmup, "predict_ms": timings, "median_ms": float(np.median(timings)),
        "boundary": "Python MLModel.predict including output materialization; excludes load, preprocessing and postprocessing.",
        "parity": parity}
    dump(args.output / "summary.json", summary)
    print(json.dumps({"median_ms": summary["median_ms"], "parity": {k:v for k,v in parity.items() if k != "predictions"}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
