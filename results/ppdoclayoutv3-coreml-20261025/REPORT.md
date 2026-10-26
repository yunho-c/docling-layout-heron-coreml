# PP-DocLayoutV3 on Core ML and Apple Neural Engine

**Feasible, with a working CPU + ANE prototype.** The fixed-800 model converts,
runs with GPU disabled, and has model-attributed ANE activity in Instruments.
Selective FP32 arithmetic is necessary for the tested graph. This is a feasibility
prototype with measured conversion drift, not a release-qualified model.

The final portable artifact is
[`optimized-mixed.mlpackage`](../../output/ppdoclayoutv3-prototype/optimized-mixed.mlpackage),
66,892,976 bytes. The locally compiled
[`optimized-mixed.mlmodelc`](../../output/ppdoclayoutv3-prototype/optimized-mixed.mlmodelc)
also passed a real-page smoke test with outputs identical to the package load.
Large artifacts, raw arrays, logs and the Instruments capture remain under the
ignored `output/ppdoclayoutv3-prototype/` directory. This report's JSON files
retain portable summaries, per-page results, placement and hashes.

## Architecture and conversion

The pinned checkpoint is
[`PaddlePaddle/PP-DocLayoutV3_safetensors`](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3_safetensors/tree/e4f489dc536556fc1be02b973ba06756e0d4a2ac),
revision `e4f489dc536556fc1be02b973ba06756e0d4a2ac`, with 33,175,165 parameters.
The implementation inspected and run was Transformers 5.8.1. V3 unifies layout
detection, instance masks and reading order; see the
[upstream architecture description](https://huggingface.co/docs/transformers/main/en/model_doc/pp_doclayout_v3).

| Component | Observed contract | Port implications |
|---|---|---|
| HGNetV2-L backbone and hybrid encoder | Three 256-channel detector maps at 100², 50² and 25²; mask features at 200² | Convolutions and feature extraction are suitable ANE work |
| Proposal selection | Top 300 from 13,125 spatial positions | Data-dependent selection and indexing; preserve query identity and ranking |
| Mask-enhanced initialization | Initial 300 masks determine decoder reference boxes | Removing mask outputs does not remove this required branch |
| Six decoder layers | Eight attention heads, three scales, four samples per scale | 18 dynamic bilinear sampling operations; native Core ML `resample` was ANE-supported on this machine |
| Final heads | 25 class logits, normalized boxes, 300×300 order logits and 300×200×200 masks | Keep sensitive small computations in FP32; dense masks create a large output |

The implemented changes are in
[`conversion/probe_doclayoutv3.py`](../../conversion/probe_doclayoutv3.py):

1. Hoist constant anchors for batch 1, 800×800 inputs.
2. Remove the size-one batch axis from deformable-attention offsets. The original
   rank-six tensor was rejected because Core ML supports rank at most five.
   Bilinear sampling, zero padding and `align_corners=False` remain unchanged.
3. Handle traced scalar casts with NumPy 2.
4. Preserve tensors shaped `[1,300,4]` and `[1,300,300]` in FP32, covering
   coordinate refinement and global-pointer scores. Other eligible operations
   use FP16. This shape-based policy is deliberately specific to this export.
5. Compute only the final class/mask/order heads while retaining all six decoder
   refinements. Intermediate auxiliary predictions do not feed later layers.
6. Compute mask bounding rectangles from row/column occupancy, avoiding large
   coordinate intermediates and an FP32-max sentinel.

The source rewrites were exactly equal to the unmodified eager model on the
first real page. Empty/full masks, all four corners and random masks also passed
exact geometry checks. TorchScript freezing itself introduced small numerical
differences, recorded in `conversion-provenance.json`; the final Core ML model
was therefore validated against the original eager model, not just the trace.

The first naive FP16 model demonstrated why conversion alone is insufficient:
CPU + ANE matched only 9/15 first-page detections at IoU ≥0.5 and produced 179
nonfinite reading-order entries. The FP32 Core ML control with `ALL` matched
15/15 with mean IoU 0.999998. Selective FP32 restored the ANE path. This isolates
a useful precision policy, but does not prove a particular individual ANE
primitive caused the original errors.

## Validation and execution evidence

The final optimized model was compared with unmodified eager FP32 PyTorch CPU
on all 50 existing IBEM sample pages. Image hashes were checked. The test keeps
the upstream processor and evaluates all layout classes, not only formulas.

| Conversion agreement measure | Result |
|---|---:|
| Reference / Core ML detections at score ≥0.30 | 1,103 / 1,105 |
| Same-class matches at IoU ≥0.50 | 1,103 |
| Missing / extra relative to FP32 | 0 / 2 |
| Mean / minimum matched box IoU | 0.985362 / 0.884146 |
| Mean matched binary-mask IoU, mask logit >0 | 0.991798 |
| Matched reading-order pair agreement | 17,485 / 17,510 (99.857%) |
| Finite outputs | All four outputs on all 50 pages |

One extra inline-formula prediction on `0005219_page07.jpg` crossed the score
cutoff: approximately 0.299608 → 0.300335. The other extra prediction on
`0009027_page01.jpg` was a high-confidence paragraph-heading duplicate:
two output boxes at scores 0.798972 and 0.795340 overlap at IoU 0.997895.
This second exception requires attention before production adoption; it is not
merely a confidence-cutoff effect. No extra NMS was introduced.

These are conversion-agreement measurements, not ground-truth layout accuracy.
Binary-mask agreement does not validate polygon extraction. Reading-order
agreement is against FP32 among matched detections, not annotated reading order.
The 50 scientific pages are an existing convenience sample, not broad coverage
of skewed scans, photographs, languages or document genres.

With `CPU_AND_NE`, the final compiler plan assigns **778 nonconstant operations
to ANE and 134 to CPU**. Estimated cost weights are 79.75% and 20.25%; these are
compiler estimates, not measured runtime shares or utilization. CPU work includes
the preserved FP32 arithmetic and some indexing/reduction operations. See
`compute-plan-cpu-ne.json` and Apple's
[compute-plan API documentation](https://apple.github.io/coremltools/source/coremltools.models.html#module-coremltools.models.compute_plan).

A separate Core AI Instruments capture ran 100 predictions with GPU disabled
and exited successfully. It recorded **1,009 ANE prediction intervals named
`optimized-mixed_main__Op0_AneInference`**, distinct from compilation/loading
events. Their union was 3,223.45 ms. This confirms actual ANE execution; it is
not an end-to-end latency or utilization measurement. Multiple ANE intervals
occur within a prediction. See `runtime-trace-summary.json`; the complete trace
and exported XML remain in the ignored output directory.

## Performance boundary

Apple M4 Pro, 48 GiB, macOS 27.0.1, AC power. Each timing run used a fresh Python
process, the same saved first-page input, three warmups and ten timed calls.
Later runs alternated `ALL` and `CPU_AND_NE`; the CPU control was measured once.
No other model experiment ran concurrently with the timed repeats. This was an
active desktop, not a controlled thermal or energy benchmark.

| Compute policy, same optimized mixed model | Warm median per process |
|---|---:|
| CPU + ANE | 49.51 ms initially; 47.16 and 46.79 ms in repeats |
| ALL, compiler selects ANE + GPU | 28.62 and 29.01 ms |
| CPU only | 85.96 ms |

The timing is the complete Python `MLModel.predict` call, including output
materialization. It excludes loading/compilation, image preprocessing and
postprocessing. This includes a 48 MB Float32 dense-mask output; it is not an
ANE-kernel timing. Loading and warmups are recorded separately in `timings.json`.
The approximately 47 ms result corresponds to about 21 sequential pages/s under
this single-input warm boundary, not sustained application throughput. The
hybrid `ALL` policy is faster in this local check; ANE remains useful when GPU
availability or resource allocation is the goal. Energy savings were not tested.

## Reproduction and remaining work

Use the existing Python 3.13 environment with Torch 2.12.0, Transformers 5.8.1,
NumPy, SciPy and Pillow. The following adds only local converter dependencies:

```sh
python3.13 -m pip install --target output/ppdoclayoutv3-prototype/python-packages \
  --no-deps coremltools==9.0 cattrs==26.2.1 pyaml==26.7.0 attrs==26.1.0

export PYTHONPATH="$PWD/output/ppdoclayoutv3-prototype/python-packages"
python3.13 conversion/probe_doclayoutv3.py --optimized --precision mixed
python3.13 conversion/validate_doclayoutv3.py \
  output/ppdoclayoutv3-prototype/optimized-mixed.mlpackage \
  --limit 50 --output output/ppdoclayoutv3-prototype/new-validation
python3.13 conversion/measure_doclayoutv3.py \
  output/ppdoclayoutv3-prototype/optimized-mixed.mlpackage \
  --units cpu-ne --output output/ppdoclayoutv3-prototype/new-measurement
xcrun coremlcompiler compile \
  output/ppdoclayoutv3-prototype/optimized-mixed.mlpackage \
  output/ppdoclayoutv3-prototype
```

The scripts use the pinned weights already in the Hugging Face cache and expose
`--dataset` where needed. They do not download weights automatically. Exact
checkpoint, source and artifact hashes are in `manifest.json`.

Input: RGB Float32 NCHW `[1,3,800,800]`, using the official processor's bicubic
resize and rescaling by 1/255, with identity mean/std. Outputs are Float32
`logits [1,300,25]`, `pred_boxes [1,300,4]` in normalized CXCYWH,
`order_logits [1,300,300]`, and `out_masks [1,300,200,200]`. Preprocessing,
thresholding, reading-order decoding and polygon extraction are external.
The model targets macOS 15 / iOS 18, but only this macOS host was tested.

The practical next milestone is to validate native integration and exact
postprocessing, investigate the duplicate-query exception and remaining order
changes, then test representative scans and target Apple devices. A reduced
output interface or selected-query mask head could lower memory traffic; the
mask-enhanced initialization must remain. Pure-ANE execution, quantization,
mobile placement, sustained throughput and power are separate unqualified work.
Coremltools 9.0 warns that Torch 2.12.0 is newer than its tested range, another
reason to retain the pinned evidence and rerun qualification on a supported
release stack before distribution.

Existing evaluation files were preserved. Model artifacts and large captures
remain local and ignored.
