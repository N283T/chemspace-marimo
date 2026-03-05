# chemspace-marimo

[![Open in molab](https://molab.marimo.io/molab-shield.svg)](https://molab.marimo.io/notebooks/nb_Sb65kt4KiKuq5cLowaaseb)
[![marimo gallery](https://img.shields.io/badge/marimo-gallery-blue)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/library/chemical-space-explorer.py)

🎉 **Featured in the [marimo official gallery](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/library/chemical-space-explorer.py)!** 🚀

Interactive chemical space viewer built with [marimo](https://marimo.io/).

Explore chemical space on a t-SNE scatter plot, tune embedding and clustering parameters, and inspect selected molecules in a reactive table.

## Demo

![Chemical Space Viewer Demo](marimo-chemspace.gif)

## Features

- **2000 NCI molecules** from RDKit built-in dataset
- **Morgan fingerprints** (ECFP4) + Tanimoto distance matrix
- **scikit-learn TSNE** 2D embedding with precomputed distances
- **HDBSCAN** density-based clustering
- **Reactive parameter controls** for t-SNE and HDBSCAN (`mo.ui.number`)
- **mo.ui.matplotlib** interactive box / lasso selection
- **Inline SVG structures** in the property table via `format_mapping`
- **Noise filtering** (`cluster = -1`) for selected rows

## Quick Start

No install needed — just run with [PEP 723](https://peps.python.org/pep-0723/) inline metadata:

```bash
uvx marimo edit ./app.py --sandbox
```

Or with a local virtual environment:

```bash
uv sync
uv run marimo edit app.py
```
