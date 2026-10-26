#!/usr/bin/env python3
"""Compare Core ML with unmodified eager FP32 V3 on real, hashed IBEM pages."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import coremltools as ct
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from measure_doclayoutv3 import compare, detections, dump
from probe_doclayoutv3 import MODEL, REVISION, OUTPUT_NAMES


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, default=Path("/Users/yunhocho/GitHub/IBEM-v1-math-detection-sample"))
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--units", choices=["cpu-ne", "all"], default="cpu-ne")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(6)
    torch.manual_seed(0)
    processor = AutoImageProcessor.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    oracle = AutoModelForObjectDetection.from_pretrained(MODEL, revision=REVISION, local_files_only=True, attn_implementation="eager").eval()
    model = ct.models.MLModel(str(args.model), compute_units=ct.ComputeUnit.CPU_AND_NE if args.units == "cpu-ne" else ct.ComputeUnit.ALL)
    records = json.loads((args.dataset / "annotations.json").read_text())
    indices = np.linspace(0, len(records)-1, args.limit, dtype=int)
    pages = []
    with torch.inference_mode():
        for index in indices:
            record = records[index]
            path = args.dataset / record["image"]
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            assert sha == record["sha256"]
            pixels = processor(images=Image.open(path).convert("RGB"), return_tensors="pt")["pixel_values"]
            result = oracle(pixel_values=pixels)
            reference = {name: getattr(result, name).numpy() for name in OUTPUT_NAMES}
            start = time.perf_counter()
            actual = model.predict({"pixel_values": pixels.numpy()})
            predict_ms = (time.perf_counter()-start)*1000
            parity = compare(reference, actual)
            rdet = detections(reference)
            rqueries = rdet["query"]
            adet = detections(actual)
            aqueries = adet["query"]
            # Small raw outputs and selected binary masks preserve reviewable
            # evidence without storing every 48 MB full-resolution mask tensor.
            np.savez_compressed(args.output / f"page-{index:02d}.npz",
                **{"reference_"+n: reference[n] for n in OUTPUT_NAMES if n != "out_masks"},
                **{"actual_"+n: actual[n] for n in OUTPUT_NAMES if n != "out_masks"},
                reference_queries=rqueries, actual_queries=aqueries,
                reference_binary_masks=reference["out_masks"][0,rqueries]>0,
                actual_binary_masks=actual["out_masks"][0,aqueries]>0)
            pages.append({"index": int(index), "image": str(path), "sha256": sha,
                "predict_ms": predict_ms, "parity": parity,
                "reference_predictions": {k:v.tolist() for k,v in rdet.items()}})
            d = parity.get("detection", {})
            print(index, path.name, {k:d[k] for k in ["matched", "missing", "extra", "mean_iou", "order_pair_count", "order_pair_agree"] if k in d}, flush=True)
            dump(args.output / "pages.json", pages)
    ds = [p["parity"]["detection"] for p in pages if "detection" in p["parity"]]
    summary = {"page_count": len(pages), "indices": indices.tolist(), "model": str(args.model), "units": args.units,
        "reference": "Unmodified eager FP32 PyTorch CPU, six threads; pinned Transformers checkpoint.",
        "all_outputs_finite": all(s["finite"] for p in pages for s in p["parity"]["raw"].values()),
        **{key: sum(d[key] for d in ds) for key in ["reference_count", "actual_count", "matched", "missing", "extra", "order_pair_count", "order_pair_agree"]},
        "mean_matched_box_iou": sum(d["mean_iou"]*d["matched"] for d in ds)/sum(d["matched"] for d in ds),
        "min_matched_box_iou": min(d["min_iou"] for d in ds),
        "mean_matched_binary_mask_iou": sum(d["mask_binary_iou_mean"]*d["matched"] for d in ds)/sum(d["matched"] for d in ds),
        "notes": ["Detection matching: native top 300 query/class pairs, score >= 0.3, same class, maximum-cardinality/IoU assignment at IoU >= 0.5.",
                  "Mask IoU is a diagnostic at mask logit > 0; it is not polygon accuracy or the processor's coupled score/mask threshold.",
                  "Reading-order pair agreement is among matched detections using the official order-sequence method; not ground-truth reading-order accuracy.",
                  "Per-page prediction times interleave oracle work and include cold first prediction; use the separate measure script for warm timing."]}
    dump(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
