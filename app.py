import marimo

__generated_with = "0.20.1"
app = marimo.App(width="full")


@app.cell
def _():
    """Shared imports."""
    import os

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from openTSNE import TSNE
    from rdkit import Chem, DataStructs, RDConfig
    from rdkit.Chem import rdFingerprintGenerator
    from sklearn.cluster import HDBSCAN

    return (
        Chem,
        DataStructs,
        HDBSCAN,
        RDConfig,
        TSNE,
        mo,
        np,
        os,
        plt,
        rdFingerprintGenerator,
    )


@app.cell
def _(Chem, DataStructs, RDConfig, np, os, rdFingerprintGenerator):
    """Load first 2000 NCI molecules, compute fingerprints and Tanimoto distance matrix."""
    from rdkit.Chem import Descriptors, Lipinski

    N_MOLS = 2000

    smi_path = os.path.join(RDConfig.RDDataDir, "NCI", "first_5K.smi")
    with open(smi_path) as f:
        lines = f.readlines()

    mols = []
    for line in lines:
        if len(mols) >= N_MOLS:
            break
        parts = line.strip().split()
        if parts:
            mol = Chem.MolFromSmiles(parts[0])
            if mol is not None:
                mols.append(mol)
    n_mols = len(mols)

    # Compute Morgan fingerprints (ECFP4)
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [morgan_gen.GetFingerprint(mol) for mol in mols]

    # Tanimoto distance matrix for t-SNE (precomputed)
    dist_matrix = np.zeros((n_mols, n_mols))
    for i in range(1, n_mols):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        for j, s in enumerate(sims):
            dist_matrix[i, j] = 1 - s
            dist_matrix[j, i] = 1 - s

    # Compute properties with RDKit descriptors
    props = []
    for mol in mols:
        props.append(
            {
                "SMILES": Chem.MolToSmiles(mol),
                "AMW": round(Descriptors.MolWt(mol), 2),
                "CLOGP": round(Descriptors.MolLogP(mol), 2),
                "HBA": Lipinski.NumHAcceptors(mol),
                "HBD": Lipinski.NumHDonors(mol),
                "Rings": Lipinski.RingCount(mol),
                "RotBonds": Lipinski.NumRotatableBonds(mol),
            }
        )
    return dist_matrix, props


@app.cell
def tsne_embedding(TSNE, dist_matrix, np):
    """Compute 2D embedding using openTSNE with precomputed Tanimoto distances."""
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        initialization="random",
        metric="precomputed",
        random_state=42,
        n_jobs=-1,
    )
    embedding = tsne.fit(dist_matrix)
    x_coords = np.array(embedding[:, 0])
    y_coords = np.array(embedding[:, 1])
    return x_coords, y_coords


@app.cell
def data_loading(mo):
    """HDBSCAN parameters."""
    min_cluster_slider = mo.ui.slider(
        start=5,
        stop=50,
        step=5,
        value=15,
        label="HDBSCAN min_cluster_size",
    )
    min_cluster_slider
    return (min_cluster_slider,)


@app.cell
def _(HDBSCAN, min_cluster_slider, np, x_coords, y_coords):
    """Cluster t-SNE coordinates with HDBSCAN."""
    coords = np.column_stack([x_coords, y_coords])
    hdb = HDBSCAN(min_cluster_size=min_cluster_slider.value, min_samples=3, copy=True)
    labels = hdb.fit_predict(coords)
    n_clusters = len(set(labels) - {-1})
    n_noise = int(np.sum(labels == -1))
    return labels, n_clusters, n_noise


@app.cell
def scatter_plot(labels, mo, n_clusters, n_noise, plt, x_coords, y_coords):
    """Interactive scatter plot colored by HDBSCAN cluster."""
    fig, ax = plt.subplots(figsize=(10, 5))

    # Noise points in gray
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

    # Clustered points
    cluster_mask = ~noise_mask
    cmap = plt.cm.get_cmap("tab20", max(n_clusters, 1))
    ax.scatter(
        x_coords[cluster_mask],
        y_coords[cluster_mask],
        c=labels[cluster_mask],
        cmap=cmap,
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


@app.cell
def selection_display(Chem, chart, labels, mo, np, props, x_coords, y_coords):
    """Scatter plot and selection table side by side."""
    from rdkit.Chem.Draw import rdMolDraw2D

    def smiles_to_svg(smiles, width=200, height=150):
        """Render SMILES as inline SVG."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return mo.Html("")
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return mo.Html(drawer.GetDrawingText())

    mask = chart.value.get_mask(x_coords, y_coords)
    selected_indices = np.where(mask)[0]

    if len(selected_indices) == 0:
        table_view = mo.md(
            "**Select molecules** on the scatter plot using box or lasso (Shift+drag)."
        )
    else:
        table_data = [{"Cluster": int(labels[i]), **props[i]} for i in selected_indices]
        table_view = mo.ui.table(
            table_data,
            format_mapping={"SMILES": smiles_to_svg},
            label=f"Selected: {len(selected_indices)} molecules",
        )

    mo.vstack([chart, table_view])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
