#!/usr/bin/env python3
"""Evaluate four detectors with a common math category and build a local viewer."""
import argparse
import collections
import json
import shutil
from pathlib import Path
import numpy as np
from evaluate_ibem import coco_ap, coverage, dump, match, rates, sha

MODELS = [
    ("heron-cpu-ne", "Heron", "Core ML FP16 · CPU + ANE · 640²"),
    ("doclayout-v2", "PP-DocLayoutV2", "Transformers FP32 · MPS GPU · 800²"),
    ("doclayout-plus-l", "PP-DocLayout_plus-L", "Transformers FP32 · MPS GPU · 800²"),
    ("doclayout-v3", "PP-DocLayoutV3", "Transformers FP32 · MPS GPU · 800²"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    annotations = json.loads((args.dataset/"annotations.json").read_text())
    manifest = json.loads((args.dataset/"manifest.json").read_text())
    assert len(annotations) == 50
    pages = []
    for p in annotations:
        assert sha(args.dataset/p["image"]) == p["sha256"]
        q = {k:p[k] for k in ["page_id","image","width","height","sha256"]}
        q["truth"] = [{"id":i,"bbox":[g[k] for k in ["x_min","y_min","x_max","y_max"]],
                       "expr_type":g["expr_type"],"latex":g["latex"]} for i,g in enumerate(p["expressions"])]
        q["models"] = {}
        pages.append(q)
    counts = dict(collections.Counter(g["expr_type"] for p in pages for g in p["truth"]))
    models = []
    for key,name,backend in MODELS:
        folder = args.output/key
        pred = {p["page_id"]:p for p in json.loads((folder/"predictions.json").read_text())}
        assert set(pred) == {p["page_id"] for p in pages}
        normalized_pages = []
        for p in pages:
            source = pred[p["page_id"]]
            assert [source["width"],source["height"]] == [p["width"],p["height"]]
            assert len({v["id"] for v in source["predictions"]}) == len(source["predictions"])
            assert all(np.isfinite(v["bbox"]).all() and np.isfinite(v["score"]) for v in source["predictions"])
            normalized_pages.append({**source,"truth":p["truth"]})
            p["models"][key] = {"predictions":source["predictions"],"matching":match(source["predictions"],p["truth"])}
        sweep = {}
        for threshold in [.1,.3,.5,.7,.9]:
            ms = [match(p["predictions"],p["truth"],threshold) for p in normalized_pages]
            metrics = rates(*(sum(m[k] for m in ms) for k in ["tp","fp","fn"]))
            type_counts = collections.Counter(g["expr_type"] for m in ms for g in m["matches"])
            metrics["recall_by_type"] = {kind:{"matched":type_counts[kind],"total":n,"recall":type_counts[kind]/n} for kind,n in counts.items()}
            metrics["matched_mean_iou"] = float(np.mean([g["iou"] for m in ms for g in m["matches"]])) if metrics["tp"] else None
            sweep[str(threshold)] = metrics
        provenance = json.loads((folder/"provenance.json").read_text())
        summary = {"key":key,"name":name,"backend":backend,"metrics":sweep["0.3"],"threshold_sweep":sweep,
                   "coco_all_math":coco_ap(normalized_pages,"all"),
                   "coverage80":{kind:coverage(normalized_pages,kind,.8) for kind in counts},
                   "diagnostic_median_forward_ms":float(np.median([v for p in normalized_pages for v in p["prediction_ms"]])),
                   "provenance":provenance}
        dump(folder/"comparison_metrics.json",summary)
        models.append(summary)
    data = {"pages":pages,"models":models,"counts":counts,"manifest":manifest,
            "metric_definition":"Formula classes collapsed to one math category; confidence > 0.30, IoU >= 0.50, confidence-ordered one-to-one per-page matching. Recall by type uses these same assignments, without requiring a predicted subtype. Precision includes all math predictions. No extra NMS.",
            "ap_definition":"pycocotools single math category, IoU 0.50:0.95, 101 recall points, maxDets 100 per page. Formula subset of native top 300 query/class predictions; no score cutoff."}
    dump(args.output/"comparison.json",data)
    (args.output/"images").mkdir(exist_ok=True)
    for p in pages:
        shutil.copy2(args.dataset/p["image"],args.output/"images"/p["image"])
    template = (Path(__file__).parent/"comparison.html").read_text()
    # UI needs no raw model/provenance payload; retain complete JSON separately.
    ui = {**data,"models":[{k:v for k,v in m.items() if k!="provenance"} for m in models]}
    (args.output/"index.html").write_text(template.replace("__DATA__",json.dumps(ui,ensure_ascii=False).replace("<","\\u003c")))
    pct = lambda x:f"{100*x:.1f}%"
    lines = ["# Four-model comparison on 50 IBEM pages", "", "[Interactive comparison](index.html) · [Complete measurements and predictions](comparison.json)", "",
             f"The sample contains **{counts['isolated']} displayed equations and {counts['embedded']} inline expressions** ({sum(counts.values())} total), from 50 distinct documents. It is the expanded deterministic convenience sample, not a random or full-test-set benchmark.", "",
             "## Shared detection results", "", data["metric_definition"], "",
             "| Model | TP / predictions | Math precision | All-math recall | Displayed recall | Inline recall | F1 | AP50 | AP50:95 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for m in models:
        s=m["metrics"];c=m["coco_all_math"];t=s["recall_by_type"]
        lines.append(f"| {m['name']} | {s['tp']} / {s['tp']+s['fp']} | {pct(s['precision'])} | {pct(s['recall'])} | {t['isolated']['matched']}/{counts['isolated']} ({pct(t['isolated']['recall'])}) | {t['embedded']['matched']}/{counts['embedded']} ({pct(t['embedded']['recall'])}) | {pct(s['f1'])} | {pct(c['ap50'])} | {pct(c['ap_50_95'])} |")
    lines += ["",data["ap_definition"],"", "The per-type columns are recall, not separate precision scores: successful inline detections are not counted as false positives in the displayed-equation column. Predicted display/inline subtype is not required to match. PP-DocLayoutV2 and V3 expose formula IDs 5 and 15 (display and inline); plus-L uses ID 7; Heron uses ID 2. Formula-number labels are excluded. Native class IDs/subtypes are retained for review.","",
              "## Confidence sensitivity", "", "The same threshold does not imply equal confidence calibration across models. No threshold was tuned on these 50 pages.", "",
              "| Model | Threshold | Precision | Recall | F1 |", "|---|---:|---:|---:|---:|"]
    for m in models:
        for threshold,s in m["threshold_sweep"].items():
            lines.append(f"| {m['name']} | >{threshold} | {pct(s['precision'])} | {pct(s['recall'])} | {pct(s['f1'])} |")
    lines += ["", "## Backend and scope", "",
              "This compares the requested deployed model paths, not architectures under an identical runtime. Heron uses the existing Core ML FP16 conversion with CPU + Neural Engine, 640×640 bilinear input. The Paddle checkpoints use pinned Hugging Face Transformers weights and their official processors, FP32 on PyTorch MPS (Mac GPU), at 800×800. Their preprocessing is not overridden to mimic Heron. V3 is scored by its bounding boxes, not segmentation polygons; reading order is not evaluated.", "",
              "Only math localization can be scored with these annotations. Other layout classes are shown for visual inspection without an accuracy score. Broad formula regions, equation-number inclusion and multiline grouping can reduce IoU against tight IBEM expression boxes. Coverage diagnostics in the JSON are many-to-one and are not detection accuracy.", "",
              "Forward timings are saved for diagnostics, but no hardware-speed ranking is claimed: precision, resolution, auxiliary networks and backend differ, and runs were not interleaved. The first DocLayoutV2 load/warmup briefly overlapped the end of the initial Heron run. Heron was rerun after all Paddle inference finished; this final isolated run is used here. The initial run is retained under `heron-cpu-ne/initial-overlap-run`.", "",
              "The earlier ten-page experiment also found different Heron boxes across Core ML compute modes. This table uses CPU + ANE only, as originally requested; it should not be read as a benchmark of upstream FP32 Heron.", "",
              "## Reproduce", "", "See [evaluation/README.md](../../evaluation/README.md). Each model directory retains raw box/logit outputs, predictions and provenance; Paddle directories include the exact runner used. Original images and annotation hashes were checked. The previous ten-page results remain intact.", "",
              "## Sources and revisions", ""]
    for m in models:
        p=m["provenance"]
        if "model" in p:
            lines.append(f"- [{p['model']}](https://huggingface.co/{p['model']}/tree/{p['revision']}), revision `{p['revision']}`.")
        else:
            lines.append(f"- Local Heron conversion at repository commit `{p['git_head']}`; model hashes recorded in provenance.")
    lines += [f"- Dataset revision `{manifest['revision']}`: {manifest['selection']}", "", "PP-FormulaNet was clarified to mean PP-DocLayout_plus-L and is not included in this comparison. No LaTeX recognition metric is claimed.", ""]
    (args.output/"REPORT.md").write_text("\n".join(lines))
    dump(args.output/"evaluation_provenance.json",{
        "annotations_sha256":sha(args.dataset/"annotations.json"),"manifest":manifest,
        "source_sha256":{str(p):sha(p) for p in Path(__file__).parent.iterdir() if p.suffix in [".py",".swift",".html"]},
        "model_metric_files":{m["key"]:sha(args.output/m["key"]/"comparison_metrics.json") for m in models}})
    print(json.dumps([{k:v for k,v in m.items() if k in ["name","metrics","coco_all_math"]} for m in models],indent=2))


if __name__ == "__main__":
    main()
