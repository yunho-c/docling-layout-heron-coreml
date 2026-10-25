#!/usr/bin/env python3
"""Run/evaluate the original Core ML Heron on annotated IBEM page images.

Uses an existing Python with numpy, Pillow and pycocotools, plus Xcode's swiftc.
No model download, retraining, image cropping, NMS, or threshold optimization.
"""
import argparse
import contextlib
import hashlib
import io
import json
import platform
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"formula": "#d64f25", "text": "#246f89", "section_header": "#725ba2",
          "title": "#a37317", "page_footer": "#666c75", "page_header": "#666c75",
          "list_item": "#29735d", "picture": "#a45982", "table": "#467f2f"}


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def match(predictions, truth, threshold=0.3, min_iou=0.5):
    """Confidence-ordered, one-to-one matching; no cross-page matching."""
    candidates = sorted([p for p in predictions if p["label"] == "formula" and p["score"] > threshold],
                        key=lambda p: -p["score"])
    used, matches, extras = set(), [], []
    for p in candidates:
        choices = [(iou(p["bbox"], g["bbox"]), i) for i, g in enumerate(truth) if i not in used]
        best_iou, index = max(choices, default=(0, -1))
        if index >= 0 and best_iou >= min_iou:
            used.add(index)
            matches.append({"prediction_id": p["id"], "truth_id": truth[index]["id"],
                            "expr_type": truth[index]["expr_type"], "iou": best_iou})
        else:
            extras.append(p["id"])
    return {"tp": len(matches), "fp": len(extras), "fn": len(truth)-len(used),
            "matches": matches, "extra_prediction_ids": extras,
            "missed_truth_ids": [g["id"] for i, g in enumerate(truth) if i not in used]}


def rates(tp, fp, fn):
    p, r = tp/(tp+fp) if tp+fp else 0, tp/(tp+fn) if tp+fn else 0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r,
            "f1": 2*p*r/(p+r) if p+r else 0}


def coverage(pages, scope, minimum):
    """Diagnostic only: many GT boxes can be covered by one predicted region."""
    hits, total = 0, 0
    for page in pages:
        predictions = [p for p in page["predictions"] if p["label"] == "formula" and p["score"] > 0.3]
        for g in page["truth"]:
            if g["expr_type"] != scope:
                continue
            total += 1
            a = g["bbox"]
            fractions = []
            for pred in predictions:
                b = pred["bbox"]
                area = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
                fractions.append(area / ((a[2]-a[0])*(a[3]-a[1])))
            hits += max(fractions, default=0) >= minimum
    return {"covered": hits, "total": total, "fraction": hits/total if total else 0}


def decode(raw, labels):
    logits = np.asarray(raw["logits"], dtype=np.float32)
    boxes = np.asarray(raw["pred_boxes"], dtype=np.float32)
    assert logits.shape == (300, 17) and boxes.shape == (300, 4)
    assert np.isfinite(logits).all() and np.isfinite(boxes).all()
    assert (boxes[:, 2:] >= 0).all()  # unused low-score FP16 queries may underflow to zero
    scores = 1 / (1 + np.exp(-logits))
    # RTDetrImageProcessor focal-loss path: top 300 across query/class pairs.
    order = np.argsort(-scores.ravel(), kind="stable")[:300]
    output = []
    for rank, flat in enumerate(order):
        q, c = divmod(int(flat), 17)
        cx, cy, w, h = map(float, boxes[q])
        output.append({"id": rank, "query_id": q, "class_id": c, "label": labels[str(c)],
                       "score": float(scores[q, c]), "bbox": [cx-w/2, cy-h/2, cx+w/2, cy+h/2]})
    return output


def coco_ap(pages, scope):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    annotations, detections, images = [], [], []
    for image_id, page in enumerate(pages):
        w, h = page["width"], page["height"]
        images.append({"id": image_id, "width": w, "height": h})
        for g in page["truth"]:
            if scope != "all" and g["expr_type"] != scope:
                continue
            x1, y1, x2, y2 = g["bbox"]
            box = [x1*w, y1*h, (x2-x1)*w, (y2-y1)*h]
            annotations.append({"id": len(annotations)+1, "image_id": image_id, "category_id": 1,
                                "bbox": box, "area": box[2]*box[3], "iscrowd": 0})
        for p in page["predictions"]:
            if p["label"] != "formula":
                continue
            x1, y1, x2, y2 = p["bbox"]
            detections.append({"image_id": image_id, "category_id": 1, "score": p["score"],
                               "bbox": [x1*w, y1*h, (x2-x1)*w, (y2-y1)*h]})
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = {"images": images, "annotations": annotations,
                      "categories": [{"id": 1, "name": "formula"}], "info": {}}
        gt.createIndex()
        dt = gt.loadRes(detections)
        ev = COCOeval(gt, dt, "bbox")
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return {"ap_50_95": float(ev.stats[0]), "ap50": float(ev.stats[1]), "ap75": float(ev.stats[2]),
            "ar_100": float(ev.stats[8]), "max_detections_per_page": 100}


def render_png(page, source, output, comparison=False):
    im = Image.open(source / page["image"]).convert("RGBA")
    w, h = im.size
    layer = Image.new("RGBA", im.size)
    draw = ImageDraw.Draw(layer)
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 17)
    if comparison:
        for g in page["truth"]:
            if g["expr_type"] != "isolated":
                continue
            box = [g["bbox"][i] * (w if i % 2 == 0 else h) for i in range(4)]
            draw.rectangle(box, outline="#168563", width=3)
    for p in reversed(page["predictions"]):
        if p["score"] <= 0.3 or (comparison and p["label"] != "formula"):
            continue
        color = COLORS.get(p["label"], "#875f3c")
        x1, y1, x2, y2 = [p["bbox"][i] * (w if i % 2 == 0 else h) for i in range(4)]
        rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
        draw.rectangle([x1,y1,x2,y2], fill=(*rgb, 28), outline=color, width=3)
        label = f'{p["label"]} {p["score"]:.2f}'
        bounds = draw.textbbox((0,0), label, font=font)
        ty = max(0, y1-23)
        draw.rectangle([x1,ty,x1+bounds[2]+10,ty+23], fill=color)
        draw.text((x1+5,ty+2), label, fill="white", font=font)
    Image.alpha_composite(im, layer).convert("RGB").save(output)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--compute-units", choices=["cpu-ne", "all", "cpu"], default="cpu-ne")
    ap.add_argument("--reuse-outputs", action="store_true")
    ap.add_argument("--metrics-only", action="store_true", help="Skip PNG overlays and HTML viewer")
    args = ap.parse_args()
    args.dataset, args.output = args.dataset.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    records = json.loads((args.dataset / "annotations.json").read_text())
    labels = json.loads((ROOT / "coreml_config.json").read_text())["id2label"]
    for p in records:
        assert sha(args.dataset / p["image"]) == p["sha256"]
        with Image.open(args.dataset / p["image"]) as im:
            assert im.size == (p["width"], p["height"])
        for g in p["expressions"]:
            assert 0 <= g["x_min"] < g["x_max"] <= 1
            assert 0 <= g["y_min"] < g["y_max"] <= 1
    if not args.reuse_outputs:
        with tempfile.TemporaryDirectory(prefix="ibem-heron-") as tmp:
            tmp = Path(tmp)
            for p in records:
                im = Image.open(args.dataset / p["image"]).convert("RGB").resize((640,640), Image.Resampling.BILINEAR)
                x = np.asarray(im, dtype=np.float32).transpose(2,0,1)[None] / 255.0
                x.astype("<f4").tofile(tmp / (p["page_id"] + ".bin"))
            binary = tmp / "RunHeron"
            subprocess.run(["xcrun", "swiftc", "-O", "-parse-as-library", str(ROOT / "evaluation/RunHeron.swift"), "-o", str(binary)], check=True)
            subprocess.run([str(binary), str(ROOT / "layout_heron.mlmodelc"), str(tmp), str(args.output), args.compute_units], check=True)
    raw = json.loads((args.output / "raw_outputs.json").read_text())
    assert raw["compute_units"] == args.compute_units
    raw_pages = {p["page_id"]: p for p in raw["pages"]}
    assert set(raw_pages) == {p["page_id"] for p in records}
    pages = []
    for p in records:
        truth = [{"id": i, "expr_type": g["expr_type"], "latex": g["latex"],
                  "bbox": [g[k] for k in ["x_min","y_min","x_max","y_max"]]} for i,g in enumerate(p["expressions"])]
        predictions = decode(raw_pages[p["page_id"]], labels)
        page = {k: p[k] for k in ["page_id","image","width","height"]}
        page.update(truth=truth, predictions=predictions, matching=match(predictions, truth),
                    prediction_ms=raw_pages[p["page_id"]]["prediction_ms"])
        pages.append(page)
    metrics = {}
    for scope in ["isolated", "embedded", "all"]:
        sweeps = {}
        for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
            results = [match(p["predictions"], [g for g in p["truth"] if scope == "all" or g["expr_type"] == scope], threshold) for p in pages]
            r = rates(*(sum(m[k] for m in results) for k in ["tp","fp","fn"]))
            r["matched_mean_iou"] = float(np.mean([v["iou"] for m in results for v in m["matches"]])) if r["tp"] else None
            sweeps[str(threshold)] = r
        metrics[scope] = {"at_score_0.3_iou_0.5": sweeps["0.3"], "score_sweep_iou_0.5": sweeps,
                          "coco": coco_ap(pages, scope)}
    times = [t for p in pages for t in p["prediction_ms"]]
    plan = json.loads((args.output / "compute_plan.json").read_text())
    provenance = {"dataset": json.loads((args.dataset / "manifest.json").read_text()),
                  "dataset_directory": str(args.dataset), "annotations_sha256": sha(args.dataset / "annotations.json"),
                  "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  "python": sys.version, "numpy": np.__version__, "pillow": PIL.__version__,
                  "macos": platform.mac_ver()[0], "chip": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
                  "power": subprocess.check_output(["pmset", "-g", "batt"], text=True).strip(),
                  "files_sha256": {str(p.relative_to(ROOT)):sha(p) for d in [ROOT/"layout_heron.mlmodelc", ROOT/"evaluation"] for p in d.rglob("*") if p.is_file() and p.suffix not in [".pyc"]},
                  "preprocessing": "Pillow RGB; bilinear stretch to 640x640; Float32 NCHW /255; no normalization or padding",
                  "postprocessing": "Sigmoid; top 300 query/class pairs per image; CXCYWH to normalized XYXY; no NMS; strict score > threshold",
                  "evaluation": "Formula predictions only, confidence-ordered one-to-one matches at IoU >= 0.5. Scope metrics use all formula predictions; predictions matching only excluded types are false positives. AP uses pycocotools, no score cutoff, maxDets=100, single category."}
    summary = {"ground_truth_counts": dict(Counter(g["expr_type"] for p in pages for g in p["truth"])),
               "page_count": len(pages), "metrics": metrics, "compute_units": raw["compute_units"], "chip": provenance["chip"],
               "compute_plan_counts": plan["nonconstant_operation_counts"],
               "timing": {"median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times,95)),
                          "min_ms": min(times), "max_ms": max(times), "count": len(times),
                          "boundary": raw["timing_boundary"], "warmup_ms": raw["warmup_ms"],
                          "model_load_ms_after_plan": raw["model_load_ms_after_plan"]}}
    summary["coverage_diagnostic"] = {
        "definition": "At score > 0.30, fraction of each GT box covered by its best-overlapping formula prediction. Many-to-one, no penalty for oversized boxes; not precision/recall or detection accuracy."
    }
    for scope in ["isolated", "embedded"]:
        summary["coverage_diagnostic"][scope] = {str(t): coverage(pages, scope, t) for t in [0.5, 0.8, 0.9]}
    dump(args.output / "summary.json", summary)
    dump(args.output / "predictions.json", pages)
    dump(args.output / "provenance.json", provenance)
    if args.metrics_only:
        print(json.dumps(summary, indent=2))
        return
    (args.output / "images").mkdir(exist_ok=True)
    (args.output / "overlays").mkdir(exist_ok=True)
    for page in pages:
        shutil.copy2(args.dataset / page["image"], args.output / "images" / page["image"])
        render_png(page, args.dataset, args.output / "overlays" / (page["page_id"]+"-layout.png"))
        render_png(page, args.dataset, args.output / "overlays" / (page["page_id"]+"-math.png"), comparison=True)
    template = (ROOT / "evaluation/viewer.html").read_text()
    data = json.dumps({"summary": summary,"pages": pages,"colors": COLORS}, ensure_ascii=False).replace("<", "\\u003c")
    (args.output / "index.html").write_text(template.replace("__DATA__", data))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
