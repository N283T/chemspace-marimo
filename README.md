# chemspace-marimo

Interactive chemical space viewer built with [marimo](https://marimo.io/).

Select molecules on a t-SNE scatter plot using box or lasso selection, and inspect their structures and properties in a reactive table.

## Demo

![Chemical Space Viewer Demo](marimo-chemspace.gif)

## Features

- **2000 NCI molecules** from RDKit built-in dataset
- **Morgan fingerprints** (ECFP4) + Tanimoto distance matrix
- **openTSNE** 2D embedding with precomputed distances
- **HDBSCAN** density-based clustering with adjustable slider
- **mo.ui.matplotlib** interactive box / lasso selection
- **Inline SVG structures** in the property table via `format_mapping`

## Quick Start

```bash
uv sync
uv run marimo edit app.py
```

## Dependencies

- marimo
- matplotlib
- rdkit
- openTSNE
- scikit-learn
