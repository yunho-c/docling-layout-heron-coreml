---
dataset_info:
  features:
  - name: image
    dtype: image
  - name: page_id
    dtype: string
  - name: expressions
    sequence:
    - name: latex
      dtype: string
    - name: latex_norm
      dtype: string
    - name: latex_expand
      dtype: string
    - name: x_min
      dtype: float32
    - name: y_min
      dtype: float32
    - name: x_max
      dtype: float32
    - name: y_max
      dtype: float32
    - name: expr_type
      dtype: string
  splits:
  - name: train
    num_bytes: 1635422184.212
    num_examples: 4649
  - name: val
    num_bytes: 308708516
    num_examples: 856
  - name: test
    num_bytes: 709594994.982
    num_examples: 1942
  download_size: 2615759043
  dataset_size: 2653725695.194
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: val
    path: data/val-*
  - split: test
    path: data/test-*
task_categories:
- object-detection
tags:
- math
- equation
- latex
size_categories:
- 1K<n<10K
---
# IBEM Full: Mathematical Expression Annotations on Scientific Pages

A structured version of the [IBEM Dataset](https://doi.org/10.5281/zenodo.7963703) (Anitei et al., 2023), designed for training and evaluation of mathematical expression detection models. This variant retains full page images and provides bounding box annotations for all mathematical expressions, enabling end-to-end layout analysis and object detection tasks.

## Dataset Summary

This dataset is derived from the original IBEM corpus by preserving full-page scientific document images and annotating all mathematical expressions using their corresponding bounding boxes and LaTeX transcriptions. It is intended for:

* Mathematical expression detection (localization and classification)
* Document understanding and layout parsing
* End-to-end systems for recognizing math expressions in context

Each record in the dataset includes:

* A full-page image from a LaTeX source document
* A unique page identifier
* A list of expressions (`expressions`), where each item contains:

  * The raw LaTeX string (`latex`)
  * A normalized version of the LaTeX string (`latex_norm`)
  * An expanded version with macros resolved (`latex_expand`)
  * Normalized bounding box coordinates (`x_min`, `y_min`, `x_max`, `y_max`) relative to image dimensions
  * The expression type (`expr_type`: either `"isolated"` or `"embedded"`)

## Dataset Structure

### Features

```python
features = Features({
    "image": Image(),
    "page_id": Value("string"),
    "expressions": Sequence({
        "latex": Value("string"),
        "latex_norm": Value("string"),
        "latex_expand": Value("string"),
        "x_min": Value("float32"),
        "y_min": Value("float32"),
        "x_max": Value("float32"),
        "y_max": Value("float32"),
        "expr_type": Value("string"),
    }),
})
```

### Splits

The dataset is split into `train`, `val`, and `test` subsets using the official partition files (`Tr*.lst`, `Va*.lst`, `Ts*.lst`) provided by the original authors.

## Source

**Citation:**

> Anitei, D., Sánchez, J. A., & Benedí, J. M. (2023). The IBEM Dataset: a large printed scientific image dataset for indexing and searching mathematical expressions (1.0) \[Data set]. Zenodo. [https://doi.org/10.5281/zenodo.7963703](https://doi.org/10.5281/zenodo.7963703)

## Original Dataset Description

The original IBEM dataset consists of 600 LaTeX documents with a total of 8,272 pages, containing:

* 29,603 isolated (displayed) mathematical expressions
* 137,089 embedded (in-line) mathematical expressions

It was created by parsing LaTeX source files from the KDD Cup Collection and supports a wide range of tasks such as:

* Mathematical expression detection and extraction
* LaTeX OCR / recognition
* Search and indexing in scientific literature

## Preprocessing Notes

* Bounding boxes are provided in relative coordinates (percentage of image width/height).
* No cropping was applied — images retain their full-page content.
* Multiple expressions per page are stored in a single list under `"expressions"`.
* Expression-level `latex`, `latex_expand`, and `latex_norm` were directly taken from the original ground truth.
* Expressions split across lines are preserved with their full content.

## Licensing

This dataset is distributed under the same license as the original IBEM dataset:
**Creative Commons Attribution 4.0 International (CC BY 4.0)**