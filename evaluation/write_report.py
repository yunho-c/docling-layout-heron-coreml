#!/usr/bin/env python3
"""Write a reviewable report and optional cross-compute-mode comparison."""
import argparse
import json
from pathlib import Path
import numpy as np
from evaluate_ibem import dump, iou, match


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("primary", type=Path)
    ap.add_argument("--reference", type=Path, action="append", default=[])
    args = ap.parse_args()
    out = args.primary
    summary = json.loads((out / "summary.json").read_text())
    pages = json.loads((out / "predictions.json").read_text())
    provenance = json.loads((out / "provenance.json").read_text())
    raw = json.loads((out / "raw_outputs.json").read_text())
    pct = lambda x: f"{100*x:.1f}%"
    rows = []
    for scope, title in [("isolated", "Displayed equations"), ("embedded", "Inline expressions"), ("all", "All math")]:
        m = summary["metrics"][scope]["at_score_0.3_iou_0.5"]
        rows.append(f"| {title} | {m['tp']} | {m['fp']} | {m['fn']} | {pct(m['precision'])} | {pct(m['recall'])} | {pct(m['f1'])} |")
    text = ["# Core ML Heron on the IBEM ten-page sample", "",
            f"Run: {raw['created_at']}. Host: {provenance['chip']}, macOS {provenance['macos']}, AC power.", "",
            "[Open the interactive bounding-box viewer](index.html). Keep the images and overlays folders beside it, or serve this directory with Python's HTTP server. Browser validation used local HTTP. All ten pages also have downloadable PNGs in `overlays/` (layout classes and math comparison).", "",
            "## Accuracy", "",
            "At **confidence > 0.30** and **IoU ≥ 0.50**, Heron produced 32 formula predictions: 12 matched displayed equations, 20 were unmatched. It matched no inline expressions. Matching is confidence-ordered, one-to-one, and confined to each page.", "",
            "| Target annotations | TP | FP | FN | Precision | Recall | F1 |",
            "|---|---:|---:|---:|---:|---:|---:|", *rows, "",
            "These are strict box-localization scores. Heron's broad formula regions often extend to equation numbers and include whitespace. IBEM uses tighter expression boxes and sometimes separates lines that Heron groups into one region. For example, `0001178_page04` has seven obvious predicted formula regions but only one reaches IoU 0.50; `0002018_page04` groups a three-line expression into one region and matches none of its five annotations at that threshold. This explains part of the score, but does not excuse missed or imprecise boxes.", "",
            "The coverage diagnostic asks whether a prediction overlaps at least 80% of each annotation's area. It covers **24/33 displayed annotations (72.7%)**, and **0/104 inline annotations**. At 50% coverage, 32/33 displayed annotations overlap a prediction. This is many-to-one, does not penalize oversized regions, and is **not detection precision/recall**.", "",
            "Only `formula` predictions are evaluated: IBEM does not provide labels for the other 16 layout classes. Non-target expression types are excluded in the separate scopes; every unmatched formula prediction still counts as FP. This is localization, not LaTeX transcription.", "",
            "### Per page", "",
            "| Page | Displayed GT | Inline GT | Formula predictions | Displayed TP | FP | Displayed FN |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for p in pages:
        gt = [g for g in p["truth"] if g["expr_type"] == "isolated"]
        m = match(p["predictions"], gt)
        text.append(f"| {p['page_id']} | {len(gt)} | {len(p['truth'])-len(gt)} | {m['tp']+m['fp']} | {m['tp']} | {m['fp']} | {m['fn']} |")
    text += ["", "### Threshold sensitivity", "", "IoU stays at 0.50. These are descriptive sweeps, not a threshold tuned on a held-out validation set.", "",
             "| Score threshold | Displayed TP | FP | Precision | Displayed recall |",
             "|---|---:|---:|---:|---:|"]
    for threshold,m in summary["metrics"]["isolated"]["score_sweep_iou_0.5"].items():
        text.append(f"| > {threshold} | {m['tp']} | {m['fp']} | {pct(m['precision'])} | {pct(m['recall'])} |")
    text += ["", "### Average precision", "", "Standard pycocotools single-class box AP, no confidence cutoff, maxDets=100 per page. Inputs are the formula subset of the detector's top 300 query/class pairs.", "",
             "| Scope | AP50 | AP75 | AP50:95 |", "|---|---:|---:|---:|"]
    for scope in ["isolated", "embedded", "all"]:
        m = summary["metrics"][scope]["coco"]
        text.append(f"| {scope} | {pct(m['ap50'])} | {pct(m['ap75'])} | {pct(m['ap_50_95'])} |")
    text += ["", "## Compute placement and timing", "",
             f"The requested **CPU + Neural Engine** mode disables the GPU. `MLComputePlan` assigns **{summary['compute_plan_counts']['ANE']:,} nonconstant operations to ANE** and **{summary['compute_plan_counts']['CPU']:,} to CPU**. The estimated cost weights are 74.8% ANE and 25.2% CPU; these are compiler estimates, not measured time or hardware utilization. This is a hybrid path, not ANE-only execution. No runtime hardware trace was collected.", "",
             f"After three warmups, five synchronous `MLModel.prediction` calls per page (50 calls total) had median **{summary['timing']['median_ms']:.2f} ms**, p95 **{summary['timing']['p95_ms']:.2f} ms**, range **{summary['timing']['min_ms']:.2f}–{summary['timing']['max_ms']:.2f} ms**. These calls exclude image decode/resize, input construction, postprocessing, and file IO. Model loading and warmups are recorded separately in `raw_outputs.json`. This is an active-host diagnostic, not a peak or end-to-end throughput benchmark."]
    comparisons = []
    if args.reference:
        text += ["", "### Same-model compute-mode checks", "",
                 "The same images and compiled model were also run under automatic and CPU-only configurations. This tests sensitivity to Core ML execution mode; it is not comparison with upstream PyTorch/ONNX.", "",
                 "| Mode | Preferred compute operations | Formula predictions | Displayed TP / 33 | Precision | Displayed recall | Warm median |",
                 "|---|---|---:|---:|---:|---:|---:|"]
        for directory in [out, *args.reference]:
            s = json.loads((directory / "summary.json").read_text())
            m = s["metrics"]["isolated"]["at_score_0.3_iou_0.5"]
            ops = ", ".join(f"{n} {d}" for d,n in s["compute_plan_counts"].items())
            text.append(f"| {s['compute_units']} | {ops} | {m['tp']+m['fp']} | {m['tp']} | {pct(m['precision'])} | {pct(m['recall'])} | {s['timing']['median_ms']:.2f} ms |")
            if directory == out:
                continue
            refs = {p["page_id"]:p for p in json.loads((directory / "predictions.json").read_text())}
            page_matches, overlaps = [], []
            for p in pages:
                a = [v for v in p["predictions"] if v["label"] == "formula" and v["score"] > .3]
                b = [v for v in refs[p["page_id"]]["predictions"] if v["label"] == "formula" and v["score"] > .3]
                used = set()
                for x in a:
                    best,j = max([(iou(x["bbox"],y["bbox"]),j) for j,y in enumerate(b) if j not in used], default=(0,-1))
                    if best >= .5:
                        overlaps.append(best); used.add(j)
                page_matches.append({"page_id":p["page_id"],"primary_count":len(a),"reference_count":len(b),"matched":len(used)})
            comparisons.append({"reference":str(directory), "mode":s["compute_units"],
                                "displayed_metrics":m,"page_matches":page_matches,
                                "matched_formula_mean_iou":float(np.mean(overlaps)),"matched_formula_min_iou":min(overlaps)})
        text += ["", "The compute modes do **not** produce interchangeable boxes on these pages. Automatic mode matched 16 displayed annotations, CPU-only 15, and CPU + ANE 12. Both precision and recall change. Root cause has not been established; this experiment does not isolate conversion error, hardware numerical behavior, or a compiler issue. Timings were collected in separate sequential runs, not interleaved controls, so treat them as descriptive only."]
    dump(out / "backend_comparison.json", comparisons)
    text += ["", "## Reproducibility and scope", "",
             f"Dataset revision: `{provenance['dataset']['revision']}`. Selection: {provenance['dataset']['selection']}", "",
             f"Model repository commit: `{provenance['git_head']}`. All model files passed the repository's `SHA256SUMS`; all ten image SHA256 values and dimensions were checked against the local annotation records. Full inputs, script fingerprints, compute plan, raw outputs, postprocessed predictions, annotations, metrics and environment details are retained in this results directory.", "",
             "Input: original JPEG converted to RGB, bilinear stretch to 640×640, Float32 NCHW divided by 255, without mean/std normalization. Output: stride-aware reads of FP16 logits/boxes; sigmoid, top 300 query/class pairs, normalized CXCYWH → XYXY. No NMS, tiling, cropping, retraining or weight modification.", "",
             "See [evaluation instructions](../../evaluation/README.md) to repeat the run. The ten-page convenience sample is insufficient for a full-dataset accuracy claim, and no accuracy qualification of the other layout classes was possible.", "",
             "Sources: [upstream preprocessing config](https://huggingface.co/docling-project/docling-layout-heron/blob/main/preprocessor_config.json), [Core ML compute-unit API](https://apple.github.io/coremltools/source/coremltools.models.html), [Apple preferred compute device semantics](https://developer.apple.com/documentation/coreml/mlcomputeplandeviceusage/preferredcomputedevice), and the [requested visualization demo](https://huggingface.co/spaces/thinkPy/Docling-Layout-Analysis). The viewer uses original code, colored overlays and class/score labels.", ""]
    (out / "REPORT.md").write_text("\n".join(text))


if __name__ == "__main__":
    main()
