# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.20.1",
#     "matplotlib>=3.10.8",
#     "pandas>=2.2.3",
#     "rdkit>=2025.9.5",
#     "scikit-learn>=1.6.1",
# ]
# ///

import marimo

__generated_with = "0.20.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    """Shared imports."""
    import os

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from rdkit import Chem, DataStructs, RDConfig, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    from sklearn.cluster import HDBSCAN
    from sklearn.manifold import TSNE
    from rdkit.Chem import Descriptors, Lipinski
    from rdkit.Chem.Draw import rdMolDraw2D

    def mol_to_svg(mol, width=200, height=150):
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return mo.Html(drawer.GetDrawingText())

    return (
        Chem,
        DataStructs,
        Descriptors,
        HDBSCAN,
        Lipinski,
        RDConfig,
        RDLogger,
        TSNE,
        mo,
        mol_to_svg,
        np,
        os,
        pd,
        plt,
        rdFingerprintGenerator,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Data Preparation
    Load NCI molecules from RDKit sample data, compute Morgan fingerprints,
    build a Tanimoto distance matrix, and assemble a `pandas.DataFrame`
    with molecular properties.
    """)
    return


@app.cell(hide_code=True)
def _(
    Chem,
    DataStructs,
    Descriptors,
    Lipinski,
    RDConfig,
    RDLogger,
    mo,
    mol_to_svg,
    np,
    os,
    pd,
    rdFingerprintGenerator,
):
    n_mols = 2000
    smi_path = os.path.join(RDConfig.RDDataDir, "NCI", "first_5K.smi")
    RDLogger.DisableLog("rdApp.error")

    supplier = Chem.SmilesMolSupplier(
        smi_path,
        delimiter="\t",
        titleLine=False,
        smilesColumn=0,
        nameColumn=1,
    )
    mols = [mol for mol in supplier if mol is not None][:n_mols]

    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [morgan_gen.GetFingerprint(mol) for mol in mols]

    dist_matrix = np.zeros((len(mols), len(mols)), dtype=np.float32)
    for i in range(1, len(mols)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dist_matrix[i, :i] = 1.0 - np.asarray(sims, dtype=np.float32)
        dist_matrix[:i, i] = dist_matrix[i, :i]

    props_df = pd.DataFrame(
        {
            "Mol": mols,
            "SMILES": [Chem.MolToSmiles(mol) for mol in mols],
            "AMW": [Descriptors.MolWt(mol) for mol in mols],
            "CLOGP": [Descriptors.MolLogP(mol) for mol in mols],
            "HBA": [Lipinski.NumHAcceptors(mol) for mol in mols],
            "HBD": [Lipinski.NumHDonors(mol) for mol in mols],
            "Rings": [Lipinski.RingCount(mol) for mol in mols],
            "RotBonds": [Lipinski.NumRotatableBonds(mol) for mol in mols],
        }
    )
    mo.ui.table(
        props_df,
        format_mapping={
            "Mol": mol_to_svg,
            "AMW": lambda x: f"{x:.2f}",
            "CLOGP": lambda x: f"{x:.2f}",
        },
        label=f"Rows : {len(props_df)}",
        page_size=5
    )
    return dist_matrix, props_df


@app.cell(hide_code=True)
def _(mo):
    perplexity_input = mo.ui.number(
        start=5, stop=80, step=1, value=30, label="t-SNE perplexity"
    )
    early_exaggeration_input = mo.ui.number(
        start=4.0,
        stop=32.0,
        step=0.5,
        value=12.0,
        label="t-SNE early_exaggeration",
    )
    learning_rate_input = mo.ui.number(
        start=10.0,
        stop=2000.0,
        step=10.0,
        value=200.0,
        label="t-SNE learning_rate",
    )
    max_iter_input = mo.ui.number(
        start=250, stop=3000, step=50, value=1000, label="t-SNE max_iter"
    )
    random_state_input = mo.ui.number(
        start=0, stop=999, step=1, value=42, label="t-SNE random_state"
    )
    min_cluster_size_input = mo.ui.number(
        start=5, stop=100, step=1, value=15, label="HDBSCAN min_cluster_size"
    )
    min_samples_input = mo.ui.number(
        start=1, stop=20, step=1, value=3, label="HDBSCAN min_samples"
    )
    include_noise_checkbox = mo.ui.checkbox(
        value=True, label="Include cluster -1 (noise)"
    )

    tsne_controls = mo.vstack(
        [
            mo.md("**t-SNE**"),
            perplexity_input,
            early_exaggeration_input,
            learning_rate_input,
            max_iter_input,
            random_state_input,
        ]
    )
    hdbscan_controls = mo.vstack(
        [mo.md("**HDBSCAN**"), min_cluster_size_input, min_samples_input]
    )
    controls = mo.hstack([tsne_controls, hdbscan_controls], align="start")
    mo.vstack([mo.md("## Embedding and Clustering"), controls])
    return (
        early_exaggeration_input,
        include_noise_checkbox,
        learning_rate_input,
        max_iter_input,
        min_cluster_size_input,
        min_samples_input,
        perplexity_input,
        random_state_input,
    )


@app.cell(hide_code=True)
def _(
    HDBSCAN,
    TSNE,
    dist_matrix,
    early_exaggeration_input,
    learning_rate_input,
    max_iter_input,
    min_cluster_size_input,
    min_samples_input,
    mo,
    np,
    perplexity_input,
    random_state_input,
):
    tsne_embedding = TSNE(
        n_components=2,
        perplexity=perplexity_input.value,
        init="random",
        metric="precomputed",
        random_state=random_state_input.value,
        learning_rate=learning_rate_input.value,
        early_exaggeration=early_exaggeration_input.value,
        max_iter=max_iter_input.value,
        n_jobs=-1,
    ).fit_transform(dist_matrix)

    labels = HDBSCAN(
        min_cluster_size=min_cluster_size_input.value,
        min_samples=min_samples_input.value,
        copy=False,
    ).fit_predict(tsne_embedding)
    n_clusters = len(set(labels) - {-1})
    n_noise = int(np.sum(labels == -1))
    x_coords, y_coords = tsne_embedding[:, 0], tsne_embedding[:, 1]

    n_total = len(labels)
    noise_ratio = (100.0 * n_noise / n_total) if n_total else 0.0
    mo.stop(
        n_total == 0,
        mo.callout(mo.md("No molecules are available for embedding."), kind="warn"),
    )
    mo.stop(
        n_clusters == 0,
        mo.callout(
            mo.md(
                f"""
                **Embedding Summary**

                No HDBSCAN clusters were found with the current parameters.

                - t-SNE perplexity: `{perplexity_input.value}`
                - t-SNE early_exaggeration: `{early_exaggeration_input.value}`
                - t-SNE learning_rate: `{learning_rate_input.value}`
                - t-SNE max_iter: `{max_iter_input.value}`
                - t-SNE random_state: `{random_state_input.value}`
                - HDBSCAN min_cluster_size: `{min_cluster_size_input.value}`
                - HDBSCAN min_samples: `{min_samples_input.value}`
                - Molecules: `{n_total}`
                - Noise points (`-1`): `{n_noise}` ({noise_ratio:.1f}%)
                """
            ),
            kind="warn",
        ),
    )
    mo.callout(
        mo.md(
            f"""
            **Embedding Summary**

            - t-SNE perplexity: `{perplexity_input.value}`
            - t-SNE early_exaggeration: `{early_exaggeration_input.value}`
            - t-SNE learning_rate: `{learning_rate_input.value}`
            - t-SNE max_iter: `{max_iter_input.value}`
            - t-SNE random_state: `{random_state_input.value}`
            - HDBSCAN min_cluster_size: `{min_cluster_size_input.value}`
            - HDBSCAN min_samples: `{min_samples_input.value}`
            - Molecules: `{n_total}`
            - Clusters (excluding `-1`): `{n_clusters}`
            - Noise points (`-1`): `{n_noise}` ({noise_ratio:.1f}%)
            """
        ),
        kind="info",
    )
    return labels, n_clusters, n_noise, x_coords, y_coords


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Selection and Table View
    Select points on the scatter plot (box/lasso with Shift+drag) to inspect
    molecule structures and descriptors in a formatted table.
    """)
    return


@app.cell(hide_code=True)
def _(labels, mo, n_clusters, n_noise, plt, x_coords, y_coords):
    fig, ax = plt.subplots(figsize=(10, 6))
    noise_mask = labels == -1
    ax.scatter(
        x_coords[noise_mask],
        y_coords[noise_mask],
        c="lightgray",
        s=15,
        alpha=0.4,
        edgecolors="none",
        label="noise",
    )
    cluster_mask = ~noise_mask
    ax.scatter(
        x_coords[cluster_mask],
        y_coords[cluster_mask],
        c=labels[cluster_mask],
        cmap=plt.get_cmap("tab20", max(n_clusters, 1)),
        s=20,
        alpha=0.8,
        edgecolors="none",
    )
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(
        f"Chemical Space — {n_clusters} clusters, {n_noise} noise (n={len(labels)})"
    )
    plt.tight_layout()
    chart = mo.ui.matplotlib(ax, debounce=True)
    return (chart,)


@app.cell(hide_code=True)
def selection_display(
    chart,
    include_noise_checkbox,
    labels,
    mo,
    mol_to_svg,
    np,
    props_df,
    x_coords,
    y_coords,
):
    selected_indices = np.where(chart.value.get_mask(x_coords, y_coords))[0]
    mo.stop(
        len(selected_indices) == 0,
        mo.vstack(
            [
                chart,
                include_noise_checkbox,
                mo.callout(
                    mo.md(
                        "Select molecules on the scatter plot using "
                        "box or lasso (Shift+drag)."
                    ),
                    kind="warn",
                ),
            ],
            align="center",
        ),
    )

    base_table = props_df.copy()
    base_table.insert(0, "Cluster", labels.astype(int))
    table_data = base_table.iloc[selected_indices].copy()
    if not include_noise_checkbox.value:
        table_data = table_data[table_data["Cluster"] != -1]

    mo.stop(
        len(table_data) == 0,
        mo.vstack(
            [
                chart,
                include_noise_checkbox,
                mo.callout(
                    mo.md(
                        "No rows to show after filtering `cluster = -1` (noise)."
                    ),
                    kind="warn",
                ),
            ],
            align="center",
        ),
    )

    table_view = mo.ui.table(
        table_data,
        format_mapping={
            "Mol": mol_to_svg,
            "AMW": lambda x: f"{x:.2f}",
            "CLOGP": lambda x: f"{x:.2f}",
        },
        label=f"Rows shown: {len(table_data)}",
    )

    mo.vstack(
        [
            chart,
            include_noise_checkbox,
            table_view,
        ],
        align="center",
    )
    return


@app.cell(hide_code=True)
def _():
    return


if __name__ == "__main__":
    app.run()
