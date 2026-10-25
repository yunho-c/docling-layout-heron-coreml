#!/usr/bin/env python3
"""Pinned Transformers PP-DocLayoutV2 inference; preserve native class IDs."""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from huggingface_hub import snapshot_download
from evaluate_ibem import dump, sha

MODELS = {
    "v2": ("PaddlePaddle/PP-DocLayoutV2_safetensors", "880e8971b88938518611c54fc0f59ad57849c9d4"),
    "plus-l": ("PaddlePaddle/PP-DocLayout_plus-L_safetensors", "325ea4e609e2db73665d41eb633413791be7d965"),
    "v3": ("PaddlePaddle/PP-DocLayoutV3_safetensors", "e4f489dc536556fc1be02b973ba06756e0d4a2ac"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", choices=["cpu", "mps"], default="mps")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--variant", choices=list(MODELS), default="v2")
    args = ap.parse_args()
    model_id, revision = MODELS[args.variant]
    args.output.mkdir(parents=True, exist_ok=True)
    records = json.loads((args.dataset / "annotations.json").read_text())
    if args.limit:
        records = records[:args.limit]
    torch.set_num_threads(6)
    torch.manual_seed(0)
    processor = AutoImageProcessor.from_pretrained(model_id, revision=revision)
    start = time.perf_counter()
    model = AutoModelForObjectDetection.from_pretrained(model_id, revision=revision).eval().to(args.device)
    sync = torch.mps.synchronize if args.device == "mps" else lambda: None
    sync()
    load_ms = (time.perf_counter()-start)*1000
    snapshot = Path(snapshot_download(model_id, revision=revision, local_files_only=True))
    native_labels = ["abstract","algorithm","aside_text","chart","content","display_formula","doc_title",
                     "figure_title","footer","footer_image","footnote","formula_number","header","header_image",
                     "image","inline_formula","number","paragraph_title","reference","reference_content","seal",
                     "table","text","vertical_text","vision_footnote"]
    if args.variant == "plus-l":
        native_labels = [model.config.id2label[i] for i in range(len(model.config.id2label))]
    (args.output / "runner.py").write_bytes(Path(__file__).read_bytes())
    pages, raw, warmups = [], [], []
    with torch.inference_mode():
        for index, record in enumerate(records):
            path = args.dataset / record["image"]
            assert sha(path) == record["sha256"]
            start = time.perf_counter()
            image = Image.open(path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(args.device)
            sync()
            preprocess_ms = (time.perf_counter()-start)*1000
            if index == 0:
                for _ in range(3):
                    sync(); start = time.perf_counter(); model(**inputs); sync()
                    warmups.append((time.perf_counter()-start)*1000)
            times = []
            for repeat in range(3):
                sync(); start = time.perf_counter(); result = model(**inputs); sync()
                times.append((time.perf_counter()-start)*1000)
                if repeat == 0:
                    logits = result.logits.detach().cpu()
                    boxes = result.pred_boxes.detach().cpu()
                    ordering = result.order_logits.detach().cpu() if hasattr(result,"order_logits") else None
                    masks = result.out_masks.detach().cpu() if hasattr(result,"out_masks") else None
            assert tuple(logits.shape) == (1,300,len(model.config.id2label))
            assert tuple(boxes.shape) == (1,300,4)
            assert torch.isfinite(logits).all() and torch.isfinite(boxes).all()
            # Official processor, with zero cutoff so AP and later score sweeps
            # are not pre-filtered by pipeline's default confidence threshold.
            from types import SimpleNamespace
            output = SimpleNamespace(logits=logits, pred_boxes=boxes, order_logits=ordering, out_masks=masks)
            post = processor.post_process_object_detection(output, threshold=0.0, target_sizes=[image.size[::-1]])[0]
            predictions = []
            orders = post.get("order_seq", torch.arange(len(post["scores"])))
            scale = torch.tensor([image.width,image.height,image.width,image.height])
            for rank, (score, label, box, order) in enumerate(zip(post["scores"],post["labels"],post["boxes"]/scale,orders)):
                label = int(label)
                predictions.append({"id":rank,"class_id":label,"label":model.config.id2label[label],
                                    "native_label":native_labels[label],"score":float(score),"bbox":box.tolist(),
                                    "reading_order":int(order)})
            predictions.sort(key=lambda p:-p["score"])
            page = {k: record[k] for k in ["page_id","image","width","height"]}
            page.update(predictions=predictions,prediction_ms=times,preprocess_ms=preprocess_ms)
            pages.append(page)
            raw.append({"page_id":record["page_id"],"logits":logits[0].tolist(),"pred_boxes":boxes[0].tolist(),
                        "order_seq":orders.tolist()})
            dump(args.output / "predictions.json", pages)
            print(f"{index+1}/{len(records)} {record['page_id']} {np.median(times):.1f} ms; "
                  f"{sum(p['label']=='formula' and p['score']>.3 for p in predictions)} math boxes",flush=True)
    dump(args.output / "raw_outputs.json",raw)
    dump(args.output / "provenance.json",{
        "model":model_id,"revision":revision,"device":args.device,"dtype":"float32",
        "transformers":transformers.__version__,"torch":torch.__version__,"macos":platform.mac_ver()[0],
        "model_load_ms":load_ms,"warmup_ms":warmups,"page_count":len(pages),
        "annotations_sha256":sha(args.dataset/"annotations.json"),
        "model_files_sha256":{p.name:sha(p) for p in snapshot.iterdir() if p.is_file()},
        "runner_sha256":sha(Path(__file__)),"processor_config":processor.to_dict(),
        "id2label":model.config.id2label,"native_label_list":native_labels,
        "formula_class_ids":[i for i,label in model.config.id2label.items() if label=="formula"],
        "postprocess":"Official HF processor threshold=0, top 300; no extra NMS. Class IDs labeled formula mapped to math; formula-number excluded. V3 evaluated by bounding boxes only; its mask/polygon predictions are not scored or displayed.",
        "timing_boundary":"Synchronized torch model forward including auxiliary networks; 3 warmups then 3 calls/page. Excludes preprocessing, device input transfer, and postprocessing. Different backend/precision/input resolution from Core ML Heron; not a controlled hardware speed comparison."
    })


if __name__ == "__main__":
    main()
