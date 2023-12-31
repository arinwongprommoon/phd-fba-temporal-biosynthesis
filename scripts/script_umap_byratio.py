#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import pickle
import umap

from matplotlib.backends.backend_pdf import PdfPages
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from src.calc.matrix import ArrayCollection
from src.calc.ablation import vget_ablation_ratio
from src.gem.yeast8model import Yeast8Model

model_options = {
    # "glc" or "pyr"
    "carbon_source": "glc",
    "num_samples": 100,
    "n_neighbors": 50,
    "min_dist": 0.5,
}


def get_random_coords(coords, num_samples):
    return coords[np.random.choice(coords.shape[0], num_samples, replace=False), :]


def coords_to_dict(coords, carbon_exch):
    """Convenience"""
    return {
        "exch_ids": [carbon_exch, "r_1654"],
        "exch_points": coords,
    }


# Initialise model
glc_exch_rate = 16.89
wt = Yeast8Model("../data/gemfiles/ecYeastGEM_batch_8-6-0.xml")
# Default: lots of glucose
wt.model.reactions.get_by_id("r_1714").bounds = (-glc_exch_rate, 0)
wt.model.reactions.get_by_id("r_1714_REV").bounds = (0, glc_exch_rate)
print("Model initialised")

# Load PKL to compute ratios
grid_filename = "ec_grid_" + model_options["carbon_source"] + "_amm"
grid_filepath = "../data/interim/" + grid_filename + ".pkl"
with open(grid_filepath, "rb") as handle:
    ablation_result_array = pickle.load(handle)

# Set axis options based on carbon source
if model_options["carbon_source"] == "glc":
    carbon_exch = "r_1714"
    saturation_glc = 8.6869
    saturation_amm = 1.4848
    x_axis = np.linspace(0, 2 * saturation_glc, 32)
elif model_options["carbon_source"] == "pyr":
    carbon_exch = "r_2033"
    saturation_pyr = 4.4444
    saturation_amm = 1.0
    x_axis = np.linspace(0, 2 * saturation_pyr, 32)
else:
    print("Error: No carbon source")
y_axis = np.linspace(0, 2 * saturation_amm, 32)

# Pick random points from a grid based on a mask,
# i.e. whether the ratio is greater than ('big') or less than ('small') one
ratio = ArrayCollection(vget_ablation_ratio(ablation_result_array), x_axis, y_axis)
ratio_array_mask = ratio.array > 1

x_coords, y_coords = np.meshgrid(x_axis, y_axis)
big_ratio_coords = np.column_stack(
    (x_coords[ratio_array_mask], y_coords[ratio_array_mask])
)
small_ratio_coords = np.column_stack(
    (x_coords[~ratio_array_mask], y_coords[~ratio_array_mask])
)

num_samples = model_options["num_samples"]

big_ratio_coords_random = get_random_coords(big_ratio_coords, num_samples)
small_ratio_coords_random = get_random_coords(small_ratio_coords, num_samples)

# Perform ablation and record fluxes
big_ablation_result_list = wt.usgfluxes_list(
    coords_to_dict(big_ratio_coords_random, carbon_exch)
)
small_ablation_result_list = wt.usgfluxes_list(
    coords_to_dict(small_ratio_coords_random, carbon_exch)
)
all_ablation_result_list = np.concatenate(
    (big_ablation_result_list, small_ablation_result_list)
)
print(f"Ablation done for {num_samples} in each category.")

# Adjust data variable dimensions
multicond_enz_use_array = np.concatenate(all_ablation_result_list)
multicond_enz_use_array.shape

# UMAP
scaled_array = scale(multicond_enz_use_array)
# TODO: Add UMAP parameters here
umapper = umap.UMAP(
    n_neighbors=model_options["n_neighbors"],
    min_dist=model_options["min_dist"],
)
embedding = umapper.fit_transform(scaled_array)
umap1 = embedding[:, 0]
umap2 = embedding[:, 1]
print("UMAP done")

# Plot each condition
num_components = 8
num_conds = int(len(umap1) / num_components)

color_dict = dict(
    zip(list(range(num_components)), ["C" + str(num) for num in range(num_components)])
)
color_list = [color_dict[el] for el in (np.arange(len(umap1)) % num_components)]
title_dict = {
    0: "ratio > 1",
    1: "ratio < 1",
}

fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(10, 5))
for cond in range(2):
    start_idx = cond * (len(umap1) // 2)
    region_range = list(range(start_idx, start_idx + (len(umap1) // 2)))
    # original, protein, carbohydrate
    # FIXME: lots of hard-cording, confusing, un-Pythonic
    # to_plot = [el for el in region_range if el % 8 in [0, 2, 3]]
    # color_list = [color_dict[el % 3] for el in range(len(to_plot))]
    to_plot = [el for el in region_range if el % 8 in [0, 1, 2, 3, 4, 5, 6, 7]]
    color_list = [color_dict[el % 8] for el in range(len(to_plot))]
    ax[cond].scatter(
        umap1[to_plot],
        umap2[to_plot],
        color=color_list,
        s=30,
        alpha=0.2,
    )
    ax[cond].set_xlim(np.min(umap1), np.max(umap1))
    ax[cond].set_ylim(np.min(umap2), np.max(umap2))
    ax[cond].tick_params(
        axis="both", bottom=False, left=False, labelbottom=False, labelleft=False
    )
    ax[cond].set_xlabel("Dimension 1")
    ax[cond].set_ylabel("Dimension 2")

    ax[cond].set_title(f"{title_dict[cond]}")

# Save all open figures to PDF
filename = "umap_" + model_options["carbon_source"] + "_byratio"
pdf_filename = "../reports/" + filename + ".pdf"
with PdfPages(pdf_filename) as pdf:
    for fig in range(1, plt.gcf().number + 1):
        pdf.savefig(fig)
# Close all figures
plt.close("all")

print("Plot drawn.")
