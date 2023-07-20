#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.backends.backend_pdf import PdfPages
from scipy.spatial.distance import pdist, squareform
from scipy.stats import zscore
from sklearn.decomposition import PCA
from src.gem.yeast8model import Yeast8Model

plot_choices = {
    "pdist": True,
    "hierarchical": False,
    "pca": True,
    "nonzero": False,
    "topflux": True,
}

model_options = {
    "glc_exch_rate": 0.194 * 8.6869,
    "pyr_exch_rate": None,
    "amm_exch_rate": 0.71 * 1.4848,
}

compute_options = {
    "zscore": False,
    "topflux/ntop": 200,
}


def prettyfloat(x):
    if x is None:
        # Unrestricted
        out_str = "Unres"
    else:
        out_str = f"{x:05.2f}".replace(".", "p")
    return out_str


def and_wrapper(a, b):
    return np.sum(np.logical_and(a, b))


def get_topn_list(series, ntop):
    """Get top N flux-carrying reactions from a Series."""
    return series.sort_values(ascending=False)[:ntop].index.to_list()


def rxns_to_hues(rxn_list, hue_lookup):
    """Convert reactions to hues"""
    hues = []
    for rxn_id in rxn_list:
        try:
            hue = hue_lookup[rxn_id]
            hues.append(hue)
        except KeyError:
            hues.append(np.nan)
    return hues


wt = Yeast8Model("../data/gemfiles/ecYeastGEM_batch_8-6-0.xml")

if model_options["glc_exch_rate"] is None:
    wt.model.reactions.get_by_id("r_1714").bounds = (-16.89, 0)
    wt.model.reactions.get_by_id("r_1714_REV").bounds = (0, 16.89)
else:
    wt.model.reactions.get_by_id("r_1714").bounds = (-model_options["glc_exch_rate"], 0)
    wt.model.reactions.get_by_id("r_1714_REV").bounds = (
        0,
        model_options["glc_exch_rate"],
    )

if model_options["pyr_exch_rate"] is None:
    pass
else:
    # wt.model.reactions.get_by_id("r_1714").bounds = (0, 0)
    wt.model.reactions.get_by_id("r_2033").bounds = (-model_options["pyr_exch_rate"], 0)
    wt.model.reactions.get_by_id("r_2033_REV").bounds = (
        0,
        model_options["pyr_exch_rate"],
    )

if model_options["amm_exch_rate"] is None:
    pass
else:
    wt.model.reactions.get_by_id("r_1654").bounds = (-model_options["amm_exch_rate"], 0)
    wt.model.reactions.get_by_id("r_1654_REV").bounds = (
        0,
        model_options["amm_exch_rate"],
    )

wt.solution = wt.optimize()
wt.ablation_result = wt.ablate()
ablation_fluxes = wt.ablation_fluxes

# Convert dictionary of pandas dataframes to numpy array for various inputs
enz_use_array = np.stack([df.to_numpy() for df in ablation_fluxes.values()])
# Remove enzymes that have all-zeros across components
# because (a) they're not informative,
# (b) they cause problems in downstream functions
enz_use_array = enz_use_array[:, np.any(enz_use_array, axis=0)]
if compute_options["zscore"]:
    # Standardise vector -- compute z-scores
    # Accounts for different dynamic ranges of fluxes for each enzyme
    enz_use_array = zscore(enz_use_array, axis=1)

list_components = list(ablation_fluxes.keys())


if plot_choices["pdist"]:
    distances = pdist(enz_use_array, metric="cosine")
    distance_matrix = squareform(distances)
    distance_triangle = np.tril(distance_matrix)
    distance_triangle[np.triu_indices(distance_triangle.shape[0])] = np.nan

    fig_pdist, ax_pdist = plt.subplots()
    sns.heatmap(
        distance_triangle,
        xticklabels=list_components,
        yticklabels=list_components,
        vmin=0,
        vmax=1,
        cmap="viridis_r",
        cbar_kws={"label": "Pairwise distances of flux vectors"},
        ax=ax_pdist,
    )

if plot_choices["hierarchical"]:
    enz_use_df = pd.DataFrame(enz_use_array, index=list_components)

    sns.clustermap(
        enz_use_df,
        cbar_kws={"label": "Fluxes"},
        col_cluster=False,
        dendrogram_ratio=0.5,
    )

if plot_choices["pca"]:
    pca = PCA()
    Xt = pca.fit_transform(enz_use_array)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    pca1 = Xt[:, 0]
    pca2 = Xt[:, 1]

    ax.scatter(pca1, pca2)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    for idx, label in enumerate(list_components):
        ax.annotate(label, (pca1[idx], pca2[idx]))

if plot_choices["nonzero"]:
    if compute_options["zscore"]:
        print("Warning: zscore option is True, nonzero analysis is invalidated.")
    enz_use_nonzero = enz_use_array != 0
    commons = pdist(enz_use_nonzero, and_wrapper)
    commons_matrix = squareform(commons)
    commons_triangle = np.tril(commons_matrix)
    commons_triangle[np.triu_indices(commons_triangle.shape[0])] = np.nan

    fig, ax = plt.subplots()
    sns.heatmap(
        commons_triangle,
        annot=True,
        fmt=".0f",
        xticklabels=list_components,
        yticklabels=list_components,
        cmap="cividis",
        cbar_kws={"label": "Number of enzymes in common with nonzero flux"},
        ax=ax,
    )

if plot_choices["topflux"]:
    ntop = compute_options["topflux/ntop"]

    # List of top N reactions, original (un-ablated)
    original_topn_list = get_topn_list(ablation_fluxes["original"], ntop)

    # Assign 'hues' and create lookup table
    hue_lookup = dict((zip(original_topn_list, range(ntop))))

    # Find hues for all components
    hues_array = []
    for series in ablation_fluxes.values():
        topn_list = get_topn_list(series, ntop)
        hues = rxns_to_hues(topn_list, hue_lookup)
        hues_array.append(hues)

    hues_array = np.array(hues_array).T

    # Visualise
    fig, ax = plt.subplots(figsize=(5, 8))
    sns.heatmap(
        hues_array,
        xticklabels=list_components,
        cmap="magma_r",
        cbar=False,
    )
    ax.set_xlabel("Biomass component")
    ax.set_ylabel("Rank")

filename = (
    "CompareEnzUse"
    + "_glc"
    + prettyfloat(model_options["glc_exch_rate"])
    + "_pyr"
    + prettyfloat(model_options["pyr_exch_rate"])
    + "_amm"
    + prettyfloat(model_options["amm_exch_rate"])
)

pdf_filename = "../reports/" + filename + ".pdf"
with PdfPages(pdf_filename) as pdf:
    for fig in range(1, plt.gcf().number + 1):
        pdf.savefig(fig)
