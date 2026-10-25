# IBEM sample evaluation

`evaluate_ibem.py` runs the existing compiled Heron artifact through native Swift
Core ML and evaluates only the `formula` class. The input dataset directory must
contain the original page JPEGs, `annotations.json` and `manifest.json`.

Use an existing Python with NumPy, Pillow and pycocotools; no virtual environment
or Core ML Python package is required. The recorded run used Python 3.13 on
macOS 27.0.1, Apple M4 Pro, with Xcode's Swift compiler.

From the repository root:

```sh
python3.13 evaluation/evaluate_ibem.py \
  --dataset /Users/yunhocho/GitHub/IBEM-v1-math-detection-sample-10 \
  --output results/ibem10-new-run
```

The default `--compute-units cpu-ne` permits CPU + Neural Engine and disables the
GPU. `all` and `cpu` are also supported. The compute plan records the compiler's
preferred devices; it is not a runtime hardware trace. The runner warms up three
times, then records five prediction calls for each page. It reads output arrays
through indexed access so padded Core ML strides are respected.

Outputs include `raw_outputs.json`, `compute_plan.json`, `predictions.json`,
`summary.json`, `provenance.json`, `index.html`, original JPEGs and PNG overlays.
Open `index.html` in a browser, or serve the results directory with
`python3.13 -m http.server --directory results/ibem10-new-run`. Browser validation
used local HTTP. PNGs use confidence > 0.30; viewer controls can change the
displayed confidence threshold. All assets are local; no remote services are used.

Use `--reuse-outputs` to regenerate evaluation/visuals from existing raw outputs
without rerunning inference. Add `--metrics-only` to skip images and HTML.
The metric uses confidence-ordered one-to-one IoU matching. COCO AP uses the
standard pycocotools evaluator without a score cutoff. Coverage diagnostics
allow many-to-one matches and must not be described as detection accuracy.

The report generator below is specific to the recorded ten-page investigation
and includes its manually reviewed observations and fixed numerical narrative:

```sh
python3.13 evaluation/write_report.py results/ibem10-20261025-cpu-ne \
  --reference results/ibem10-20261025-all \
  --reference results/ibem10-20261025-cpu
```

No model weights or original dataset files are modified. The generated results
remain local; this workflow does not publish or upload the samples.

## Four-model comparison on the expanded 50-page sample

The user clarified `PP-FormulaNet_plus-L` to mean `PP-DocLayout_plus-L` and added
PP-DocLayoutV3. The final comparison includes four **detectors**, with no LaTeX
recognition score. The small earlier FormulaNet interface pilot is not included.

The Paddle runner uses the existing Python 3.13 environment (Transformers 5.8.1,
PyTorch 2.12.0, torchvision, OpenCV, NumPy, Pillow, pycocotools) and downloads
pinned safetensors checkpoints into the normal Hugging Face cache. No new virtual
environment is needed. Run inference sequentially to avoid timing interference:

```sh
python3.13 evaluation/evaluate_ibem.py \
  --dataset /Users/yunhocho/GitHub/IBEM-v1-math-detection-sample \
  --output results/ibem50-20261025/heron-cpu-ne --metrics-only

python3.13 evaluation/run_doclayout.py \
  --dataset /Users/yunhocho/GitHub/IBEM-v1-math-detection-sample \
  --output results/ibem50-20261025/doclayout-v2 --variant v2

python3.13 evaluation/run_doclayout.py \
  --dataset /Users/yunhocho/GitHub/IBEM-v1-math-detection-sample \
  --output results/ibem50-20261025/doclayout-plus-l --variant plus-l

python3.13 evaluation/run_doclayout.py \
  --dataset /Users/yunhocho/GitHub/IBEM-v1-math-detection-sample \
  --output results/ibem50-20261025/doclayout-v3 --variant v3

python3.13 evaluation/build_comparison.py \
  --dataset /Users/yunhocho/GitHub/IBEM-v1-math-detection-sample \
  --output results/ibem50-20261025

python3.13 -m http.server 8767 --bind 127.0.0.1 \
  --directory results/ibem50-20261025
```

The comparison uses a single math category and reports recall by the annotation's
displayed/inline type from the same one-to-one matching. This avoids treating a
correct inline detection as a false positive in a displayed-only precision score.
Formula-number classes are excluded. Official model preprocessing is preserved;
the shared score threshold does not imply shared confidence calibration.

The viewer supports any model pair, synchronized scrolling, confidence changes,
annotation toggles, matched/missed overlays, page filtering, box inspection, and
PNG export. The fixed summary table stays at score >0.30, IoU >=0.50. COCO AP is
computed without a score cutoff. V3 polygons and reading order are not evaluated.

`run_doclayout.py` saves native class IDs and subtypes; both IDs 5 and 15 of V2/V3
map to math. Its V3 postprocessor runs at zero score threshold to retain candidates
for AP; only returned **boxes and scores** are used, not the polygons whose mask
threshold also depends on that argument.
