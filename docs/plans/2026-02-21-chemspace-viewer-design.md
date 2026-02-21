# Chemical Space Viewer Design

Date: 2026-02-21

## Goal

Try `mo.ui.matplotlib` reactive box/lasso selection for chemical space visualization.

## Architecture

Single marimo notebook (`app.py`) with 4 cells:

| Cell | Responsibility |
|------|---------------|
| 1 | Data loading: `first_200.props.sdf` -> Mol list + Morgan FP (ECFP4, radius=2, 2048bit) |
| 2 | Dimensionality reduction: openTSNE 2D embedding |
| 3 | Scatter plot: `mo.ui.matplotlib` with CLOGP color mapping + box/lasso selection |
| 4 | Selection display: molecule structure images (`MolsToGridImage`) + property table |

## Data Flow

```
SDF -> RDKit Mol -> Morgan FP (radius=2, 2048bit)
                 -> openTSNE -> (x, y) coordinates
                 -> matplotlib scatter -> mo.ui.matplotlib
                                           | selection
                 -> get_mask() -> selected molecule indices
                 -> structure images + property table
```

## Decisions

- **Data source**: RDKit built-in `first_200.props.sdf` (200 molecules with rich properties)
- **Fingerprint**: Morgan (ECFP4), radius=2, nBits=2048
- **Dim reduction**: openTSNE only (no PCA/UMAP)
- **Color**: CLOGP (continuous value, intuitive for chemists)
- **Selection output**: Both structure images and property table (SMILES, AMW, CLOGP, Lipinski violations, etc.)

---
- [ ] **DONE** - Design complete
