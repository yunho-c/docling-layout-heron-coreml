---
license: apache-2.0
library_name: coreml
pipeline_tag: object-detection
base_model: docling-project/docling-layout-heron
tags:
  - coreml
  - coremltools
  - document-layout-analysis
  - document-ai
  - rt-detr-v2
  - docling
---

# docling-layout-heron for Core ML

Core ML conversion of
[`docling-project/docling-layout-heron`](https://huggingface.co/docling-project/docling-layout-heron),
Docling's RT-DETRv2 document-layout model. The weights were converted, not
retrained or fine-tuned.

This repository contains two equivalent artifacts:

- `layout_heron.mlpackage` — portable Core ML ML Program (recommended for
  redistribution and inspection).
- `layout_heron.mlmodelc` — precompiled copy for direct loading by
  [`docling.rs`](https://github.com/docling-project/docling.rs) and other native
  Core ML clients. It was compiled with Xcode 26.6.

The model requires macOS 15+, iOS/iPadOS 18+, macCatalyst 18+, tvOS 18+,
visionOS 2+, or watchOS 11+. It is an ML Program with FP16 weight and compute
precision, a fixed batch size of one, and raw detector outputs.

## Input and output contract

Input:

- `pixel_values`: Float32 `[1, 3, 640, 640]`, NCHW RGB.
- Stretch the page image to exactly 640 × 640 using bilinear resampling.
- Rescale bytes to `[0, 1]` by multiplying by `1/255`.
- Do not letterbox, pad, or apply mean/std normalization.

Outputs:

- `logits`: Float16 `[1, 300, 17]` raw class logits.
- `pred_boxes`: Float16 `[1, 300, 4]` normalized `(center_x, center_y,
  width, height)` boxes.

Those are the model-schema dtypes. `coremltools` may expose both arrays as
Float32 NumPy values after prediction.

The output is intentionally raw. Apply sigmoid to `logits`, choose the highest
query/class scores, threshold them, convert boxes from normalized CXCYWH to XYXY,
and scale them back to the original page dimensions. The 17 labels and all
machine-readable shape metadata are in `coreml_config.json`.

## Use with docling.rs

Place either artifact at the default path:

```bash
mkdir -p .models
cp -R layout_heron.mlmodelc .models/layout_heron.mlmodelc
```

Then run a normal macOS build of `docling.rs`. Its default `coreml-native`
feature automatically prefers the native Core ML layout backend when the model
is present. You can also set an explicit path:

```bash
export DOCLING_RS_LAYOUT_ENGINE=coreml
export DOCLING_LAYOUT_COREML="$PWD/layout_heron.mlpackage"
cargo run -p docling-cli -- document.pdf
```

`DOCLING_RS_LAYOUT_ENGINE=coreml` means the native Core ML backend. It is
different from the ONNX Runtime CoreML execution-provider feature.

## Use from Python

Core ML execution requires macOS:

```python
import coremltools as ct
import numpy as np
from PIL import Image

model = ct.models.MLModel("layout_heron.mlpackage", compute_units=ct.ComputeUnit.ALL)

image = Image.open("page.png").convert("RGB").resize((640, 640), Image.Resampling.BILINEAR)
pixel_values = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0

output = model.predict({"pixel_values": pixel_values})
logits = output["logits"]
pred_boxes = output["pred_boxes"]
```

## Validation

The conversion was checked against the FP32 ONNX export on 12 inputs: nine real
document pages and three uniform-random tensors. At a 0.3 score threshold,
greedy same-class matching with IoU > 0.5 produced:

| Backend | Matched | Missing vs FP32 | Extra vs FP32 | Mean IoU | Min IoU |
|---|---:|---:|---:|---:|---:|
| Core ML FP16 | 239 | 0 | 0 | 0.9886 | 0.8283 |
| ONNX INT8 | 224 | 15 | 11 | 0.9813 | 0.6798 |

On the conversion machine (Apple Silicon, macOS 26.5), median batch-1 latency
over 20 warm runs was 12.31 ms with `ComputeUnit.ALL`, compared with 237.06 ms
for ONNX FP32 CPU and 161.73 ms for ONNX INT8 CPU. These are conversion checks,
not broad benchmark claims; performance varies by device and OS.

The Core ML compute plan placed all 2,427 non-constant compute operations on the
GPU with `ComputeUnit.ALL`. `CPU_AND_NE` was slower for this graph.

## IBEM comparison experiment

The experimental IBEM evaluation compares this Core ML model with
PP-DocLayoutV2, PP-DocLayout_plus-L, and PP-DocLayoutV3 on 50 scientific pages.
See the [comparison report](results/ibem50-20261025/REPORT.md),
[interactive viewer](results/ibem50-20261025/index.html), and
[evaluation instructions](evaluation/README.md). Saved results include raw
outputs, model revisions, compute placement, and dataset provenance. The
comparison uses different runtimes and input resolutions and is not a
full-dataset benchmark.

## Reproduce the conversion

The conversion requires macOS and Python 3.12:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r conversion/requirements.txt
.venv/bin/python conversion/convert.py
xcrun coremlcompiler compile layout_heron.mlpackage .
```

`conversion/convert.py` contains the three compatibility rewrites needed by
coremltools 8.3 for RT-DETRv2: bilinear `grid_sample` decomposition, scalar
casting for a one-element traced tensor, and an out-of-place anchor-generation
rewrite. The rewrites are numerically equivalent for this fixed inference
contract.

## Limitations

- Fixed batch size 1 and fixed 640 × 640 input.
- Raw detector outputs; preprocessing and postprocessing are not embedded.
- The precompiled `.mlmodelc` was built by Xcode 26.6. If it does not load on a
  target system, use the `.mlpackage` so Core ML can compile it locally.
- The validation set is small and is intended to catch conversion drift, not to
  re-evaluate the upstream model's accuracy or fairness.
- This conversion inherits the upstream model's intended uses, limitations,
  training data, and risks. See the
  [upstream model card](https://huggingface.co/docling-project/docling-layout-heron)
  and [technical report](https://arxiv.org/abs/2509.11720).

## License and attribution

The upstream model is Apache-2.0 licensed. This repository is a format
conversion and retains that license and attribution. See `LICENSE` and `NOTICE`.

If you use the model, please cite the upstream Heron and Docling work listed in
the upstream model card.
