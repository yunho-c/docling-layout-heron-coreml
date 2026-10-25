# Four-model comparison on 50 IBEM pages

[Interactive comparison](index.html) · [Complete measurements and predictions](comparison.json)

The sample contains **96 displayed equations and 715 inline expressions** (811 total), from 50 distinct documents. It is the expanded deterministic convenience sample, not a random or full-test-set benchmark.

## Shared detection results

Formula classes collapsed to one math category; confidence > 0.30, IoU >= 0.50, confidence-ordered one-to-one per-page matching. Recall by type uses these same assignments, without requiring a predicted subtype. Precision includes all math predictions. No extra NMS.

| Model | TP / predictions | Math precision | All-math recall | Displayed recall | Inline recall | F1 | AP50 | AP50:95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Heron | 44 / 97 | 45.4% | 5.4% | 44/96 (45.8%) | 0/715 (0.0%) | 9.7% | 4.5% | 1.5% |
| PP-DocLayoutV2 | 571 / 700 | 81.6% | 70.4% | 96/96 (100.0%) | 475/715 (66.4%) | 75.6% | 74.2% | 34.7% |
| PP-DocLayout_plus-L | 549 / 649 | 84.6% | 67.7% | 93/96 (96.9%) | 456/715 (63.8%) | 75.2% | 76.1% | 32.7% |
| PP-DocLayoutV3 | 485 / 530 | 91.5% | 59.8% | 93/96 (96.9%) | 392/715 (54.8%) | 72.3% | 73.6% | 36.9% |

pycocotools single math category, IoU 0.50:0.95, 101 recall points, maxDets 100 per page. Formula subset of native top 300 query/class predictions; no score cutoff.

The per-type columns are recall, not separate precision scores: successful inline detections are not counted as false positives in the displayed-equation column. Predicted display/inline subtype is not required to match. PP-DocLayoutV2 and V3 expose formula IDs 5 and 15 (display and inline); plus-L uses ID 7; Heron uses ID 2. Formula-number labels are excluded. Native class IDs/subtypes are retained for review.

## Confidence sensitivity

The same threshold does not imply equal confidence calibration across models. No threshold was tuned on these 50 pages.

| Model | Threshold | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Heron | >0.1 | 30.5% | 6.3% | 10.4% |
| Heron | >0.3 | 45.4% | 5.4% | 9.7% |
| Heron | >0.5 | 48.4% | 5.4% | 9.8% |
| Heron | >0.7 | 51.2% | 5.2% | 9.4% |
| Heron | >0.9 | 52.8% | 2.3% | 4.5% |
| PP-DocLayoutV2 | >0.1 | 50.9% | 80.3% | 62.3% |
| PP-DocLayoutV2 | >0.3 | 81.6% | 70.4% | 75.6% |
| PP-DocLayoutV2 | >0.5 | 91.5% | 60.8% | 73.0% |
| PP-DocLayoutV2 | >0.7 | 94.7% | 52.8% | 67.8% |
| PP-DocLayoutV2 | >0.9 | 94.7% | 28.5% | 43.8% |
| PP-DocLayout_plus-L | >0.1 | 56.5% | 81.0% | 66.6% |
| PP-DocLayout_plus-L | >0.3 | 84.6% | 67.7% | 75.2% |
| PP-DocLayout_plus-L | >0.5 | 93.3% | 59.8% | 72.9% |
| PP-DocLayout_plus-L | >0.7 | 96.5% | 40.8% | 57.4% |
| PP-DocLayout_plus-L | >0.9 | 98.0% | 12.2% | 21.7% |
| PP-DocLayoutV3 | >0.1 | 78.2% | 75.2% | 76.7% |
| PP-DocLayoutV3 | >0.3 | 91.5% | 59.8% | 72.3% |
| PP-DocLayoutV3 | >0.5 | 95.6% | 53.3% | 68.4% |
| PP-DocLayoutV3 | >0.7 | 95.9% | 40.2% | 56.6% |
| PP-DocLayoutV3 | >0.9 | 97.8% | 5.4% | 10.3% |

## Backend and scope

This compares the requested deployed model paths, not architectures under an identical runtime. Heron uses the existing Core ML FP16 conversion with CPU + Neural Engine, 640×640 bilinear input. The Paddle checkpoints use pinned Hugging Face Transformers weights and their official processors, FP32 on PyTorch MPS (Mac GPU), at 800×800. Their preprocessing is not overridden to mimic Heron. V3 is scored by its bounding boxes, not segmentation polygons; reading order is not evaluated.

Only math localization can be scored with these annotations. Other layout classes are shown for visual inspection without an accuracy score. Broad formula regions, equation-number inclusion and multiline grouping can reduce IoU against tight IBEM expression boxes. Coverage diagnostics in the JSON are many-to-one and are not detection accuracy.

Forward timings are saved for diagnostics, but no hardware-speed ranking is claimed: precision, resolution, auxiliary networks and backend differ, and runs were not interleaved. The first DocLayoutV2 load/warmup briefly overlapped the end of the initial Heron run. Heron was rerun after all Paddle inference finished; this final isolated run is used here. The initial run is retained under `heron-cpu-ne/initial-overlap-run`.

The earlier ten-page experiment also found different Heron boxes across Core ML compute modes. This table uses CPU + ANE only, as originally requested; it should not be read as a benchmark of upstream FP32 Heron.

## Reproduce

See [evaluation/README.md](../../evaluation/README.md). Each model directory retains raw box/logit outputs, predictions and provenance; Paddle directories include the exact runner used. Original images and annotation hashes were checked. The previous ten-page results remain intact.

## Sources and revisions

- Local Heron conversion at repository commit `3d4f06aa152a72b05f8d47f4d29e2906eb47aa80`; model hashes recorded in provenance.
- [PaddlePaddle/PP-DocLayoutV2_safetensors](https://huggingface.co/PaddlePaddle/PP-DocLayoutV2_safetensors/tree/880e8971b88938518611c54fc0f59ad57849c9d4), revision `880e8971b88938518611c54fc0f59ad57849c9d4`.
- [PaddlePaddle/PP-DocLayout_plus-L_safetensors](https://huggingface.co/PaddlePaddle/PP-DocLayout_plus-L_safetensors/tree/325ea4e609e2db73665d41eb633413791be7d965), revision `325ea4e609e2db73665d41eb633413791be7d965`.
- [PaddlePaddle/PP-DocLayoutV3_safetensors](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3_safetensors/tree/e4f489dc536556fc1be02b973ba06756e0d4a2ac), revision `e4f489dc536556fc1be02b973ba06756e0d4a2ac`.
- Dataset revision `4f82284a793f020a9235efe26dbd6ed5d733a022`: First page encountered from each of the first 50 distinct documents in the first test shard. Preserves the initial ten samples and adds forty distinct documents. Deterministic convenience sample, not a random sample.

PP-FormulaNet was clarified to mean PP-DocLayout_plus-L and is not included in this comparison. No LaTeX recognition metric is claimed.
