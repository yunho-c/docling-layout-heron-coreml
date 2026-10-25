# Core ML Heron on the IBEM ten-page sample

Run: 2026-10-25T23:00:16Z. Host: Apple M4 Pro, macOS 27.0.1, AC power.

[Open the interactive bounding-box viewer](index.html). Keep the images and overlays folders beside it, or serve this directory with Python's HTTP server. Browser validation used local HTTP. All ten pages also have downloadable PNGs in `overlays/` (layout classes and math comparison).

## Accuracy

At **confidence > 0.30** and **IoU ≥ 0.50**, Heron produced 32 formula predictions: 12 matched displayed equations, 20 were unmatched. It matched no inline expressions. Matching is confidence-ordered, one-to-one, and confined to each page.

| Target annotations | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Displayed equations | 12 | 20 | 21 | 37.5% | 36.4% | 36.9% |
| Inline expressions | 0 | 32 | 104 | 0.0% | 0.0% | 0.0% |
| All math | 12 | 20 | 125 | 37.5% | 8.8% | 14.2% |

These are strict box-localization scores. Heron's broad formula regions often extend to equation numbers and include whitespace. IBEM uses tighter expression boxes and sometimes separates lines that Heron groups into one region. For example, `0001178_page04` has seven obvious predicted formula regions but only one reaches IoU 0.50; `0002018_page04` groups a three-line expression into one region and matches none of its five annotations at that threshold. This explains part of the score, but does not excuse missed or imprecise boxes.

The coverage diagnostic asks whether a prediction overlaps at least 80% of each annotation's area. It covers **24/33 displayed annotations (72.7%)**, and **0/104 inline annotations**. At 50% coverage, 32/33 displayed annotations overlap a prediction. This is many-to-one, does not penalize oversized regions, and is **not detection precision/recall**.

Only `formula` predictions are evaluated: IBEM does not provide labels for the other 16 layout classes. Non-target expression types are excluded in the separate scopes; every unmatched formula prediction still counts as FP. This is localization, not LaTeX transcription.

### Per page

| Page | Displayed GT | Inline GT | Formula predictions | Displayed TP | FP | Displayed FN |
|---|---:|---:|---:|---:|---:|---:|
| 0001074_page02 | 0 | 8 | 0 | 0 | 0 | 0 |
| 0001125_page03 | 6 | 15 | 6 | 5 | 1 | 1 |
| 0001133_page03 | 0 | 9 | 0 | 0 | 0 | 0 |
| 0001178_page04 | 7 | 19 | 7 | 1 | 6 | 6 |
| 0002018_page04 | 5 | 13 | 3 | 0 | 3 | 5 |
| 0002177_page01 | 0 | 3 | 0 | 0 | 0 | 0 |
| 0003055_page01 | 0 | 12 | 0 | 0 | 0 | 0 |
| 0003079_page01 | 0 | 5 | 0 | 0 | 0 | 0 |
| 0003126_page03 | 15 | 15 | 16 | 6 | 10 | 9 |
| 0003226_page01 | 0 | 5 | 0 | 0 | 0 | 0 |

### Threshold sensitivity

IoU stays at 0.50. These are descriptive sweeps, not a threshold tuned on a held-out validation set.

| Score threshold | Displayed TP | FP | Precision | Displayed recall |
|---|---:|---:|---:|---:|
| > 0.1 | 15 | 32 | 31.9% | 45.5% |
| > 0.3 | 12 | 20 | 37.5% | 36.4% |
| > 0.5 | 12 | 18 | 40.0% | 36.4% |
| > 0.7 | 12 | 16 | 42.9% | 36.4% |
| > 0.9 | 4 | 4 | 50.0% | 12.1% |

### Average precision

Standard pycocotools single-class box AP, no confidence cutoff, maxDets=100 per page. Inputs are the formula subset of the detector's top 300 query/class pairs.

| Scope | AP50 | AP75 | AP50:95 |
|---|---:|---:|---:|
| isolated | 28.6% | 2.8% | 6.7% |
| embedded | 0.0% | 0.0% | 0.0% |
| all | 7.0% | 0.7% | 1.7% |

## Compute placement and timing

The requested **CPU + Neural Engine** mode disables the GPU. `MLComputePlan` assigns **1,052 nonconstant operations to ANE** and **1,375 to CPU**. The estimated cost weights are 74.8% ANE and 25.2% CPU; these are compiler estimates, not measured time or hardware utilization. This is a hybrid path, not ANE-only execution. No runtime hardware trace was collected.

After three warmups, five synchronous `MLModel.prediction` calls per page (50 calls total) had median **71.07 ms**, p95 **72.70 ms**, range **69.56–75.98 ms**. These calls exclude image decode/resize, input construction, postprocessing, and file IO. Model loading and warmups are recorded separately in `raw_outputs.json`. This is an active-host diagnostic, not a peak or end-to-end throughput benchmark.

### Same-model compute-mode checks

The same images and compiled model were also run under automatic and CPU-only configurations. This tests sensitivity to Core ML execution mode; it is not comparison with upstream PyTorch/ONNX.

| Mode | Preferred compute operations | Formula predictions | Displayed TP / 33 | Precision | Displayed recall | Warm median |
|---|---|---:|---:|---:|---:|---:|
| cpu-ne | 1052 ANE, 1375 CPU | 32 | 12 | 37.5% | 36.4% | 71.07 ms |
| all | 259 ANE, 2168 GPU | 31 | 16 | 51.6% | 48.5% | 28.70 ms |
| cpu | 2427 CPU | 35 | 15 | 42.9% | 45.5% | 89.84 ms |

The compute modes do **not** produce interchangeable boxes on these pages. Automatic mode matched 16 displayed annotations, CPU-only 15, and CPU + ANE 12. Both precision and recall change. Root cause has not been established; this experiment does not isolate conversion error, hardware numerical behavior, or a compiler issue. Timings were collected in separate sequential runs, not interleaved controls, so treat them as descriptive only.

## Reproducibility and scope

Dataset revision: `4f82284a793f020a9235efe26dbd6ed5d733a022`. Selection: First page encountered from each of the first ten distinct documents in row group 0. Deterministic convenience sample, not a random sample.

Model repository commit: `3d4f06aa152a72b05f8d47f4d29e2906eb47aa80`. All model files passed the repository's `SHA256SUMS`; all ten image SHA256 values and dimensions were checked against the local annotation records. Full inputs, script fingerprints, compute plan, raw outputs, postprocessed predictions, annotations, metrics and environment details are retained in this results directory.

Input: original JPEG converted to RGB, bilinear stretch to 640×640, Float32 NCHW divided by 255, without mean/std normalization. Output: stride-aware reads of FP16 logits/boxes; sigmoid, top 300 query/class pairs, normalized CXCYWH → XYXY. No NMS, tiling, cropping, retraining or weight modification.

See [evaluation instructions](../../evaluation/README.md) to repeat the run. The ten-page convenience sample is insufficient for a full-dataset accuracy claim, and no accuracy qualification of the other layout classes was possible.

Sources: [upstream preprocessing config](https://huggingface.co/docling-project/docling-layout-heron/blob/main/preprocessor_config.json), [Core ML compute-unit API](https://apple.github.io/coremltools/source/coremltools.models.html), [Apple preferred compute device semantics](https://developer.apple.com/documentation/coreml/mlcomputeplandeviceusage/preferredcomputedevice), and the [requested visualization demo](https://huggingface.co/spaces/thinkPy/Docling-Layout-Analysis). The viewer uses original code, colored overlays and class/score labels.
