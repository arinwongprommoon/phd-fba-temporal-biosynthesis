#!/usr/bin/env python3
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

from src.calc.ablation import vget_ablation_ratio
from src.viz.grid import heatmap_ablation_grid

model_options = {
    # "ec" or "y8"
    "model": "ec",
    # "glc" or "pyr"
    "carbon_source": "glc",
}

axis_options = {
    "ratio_vmin": 0.5,
    "ratio_vmax": 1.5,
    "grid_xlabel_leader": None,
    "grid_ylabel_leader": "Ammonium exchange",
}

plot_choices = {
    "heatmap_ratio": True,
    "heatmap_growthrate": True,
    "scatter_growthrate_ratio": True,
    "heatmap_gradient_c": True,
    "heatmap_gradient_n": True,
    "heatmap_gradient_compare": True,
    "heatmap_ratio_whereone": True,
    "heatmap_carb_to_prot": True,
    "histogram_carb_to_prot": True,
}


@np.vectorize
def vget_growthrate(x):
    return x.ablated_flux[0]


@np.vectorize
def vget_carb_to_prot_ratio(x):
    carb_to_prot = x.ablated_est_time[3] / x.ablated_est_time[2]
    return carb_to_prot


if model_options["model"] == "ec":
    exch_rate_dict = {
        "r_1714": np.linspace(0, 2 * 8.45, 32),  # glucose
        "r_2033": np.linspace(0, 2 * 4.27, 32),  # pyruvate
        "r_1654": np.linspace(0, 2 * 1.45, 32),  # ammonium
    }
elif model_options["model"] == "y8":
    exch_rate_dict = {
        "r_1714": np.linspace(0, 2 * 4.75, 32),  # glucose
        "r_2033": np.linspace(0, 2 * 13.32, 32),  # pyruvate
        "r_1654": np.linspace(0, 2 * 2.88, 32),  # ammonium
    }
else:
    m = model_options["model"]
    print(f"Invalid model {m}")

if model_options["carbon_source"] == "glc":
    exch_rate_dict.pop("r_2033")
    axis_options["grid_xlabel_leader"] = "Glucose exchange"
elif model_options["carbon_source"] == "pyr":
    exch_rate_dict.pop("r_1714")
    axis_options["grid_xlabel_leader"] = "Pyruvate exchange"

# Load saved data
filename = model_options["model"] + "_grid_" + model_options["carbon_source"] + "_amm"
filepath = "../data/interim/" + filename + ".pkl"
with open(filepath, "rb") as handle:
    ablation_result_array = pickle.load(handle)

# Compute data
# Generate numpy arrays from ablation_result_array
ratio_array = vget_ablation_ratio(ablation_result_array)
growthrate_array = vget_growthrate(ablation_result_array)
growthrate_gradient = np.gradient(growthrate_array)
growthrate_gradient_greater = np.abs(growthrate_gradient[0]) - np.abs(
    growthrate_gradient[1]
)
ratio_array_mask = ratio_array > 1
carb_to_prot_array = vget_carb_to_prot_ratio(ablation_result_array)

# Prepare lists for ratio vs growthrate plot
ratios = ratio_array[1:, 1:].ravel()
growthrates = growthrate_array[1:, 1:].ravel()

# Prepare data structures for carb:prot ratio vs abl ratio plots
carb_to_prot_ratios = carb_to_prot_array[1:, 1:].ravel()
ratio_bools = ratio_array_mask[1:, 1:].ravel()

carb_to_prot_df = pd.DataFrame(
    {
        "carb_to_prot_ratio": carb_to_prot_ratios,
        "ratio_bool": ratio_bools,
    }
)

# Set up axes parameters
xmax = np.max(list(exch_rate_dict.values())[0])
ymax = np.max(list(exch_rate_dict.values())[0])
grid_xlabel_leader = axis_options["grid_xlabel_leader"]
grid_ylabel_leader = axis_options["grid_ylabel_leader"]
grid_xlabel = f"{grid_xlabel_leader} (% max = {xmax:.2f})"
grid_ylabel = f"{grid_ylabel_leader} (% max = {ymax:.2f})"

# Construct dict that tells which ax to draw names plots in
plot_axs_keys = list(plot_choices.keys())
plot_axs_values = []
idx = 0
for plot_choice in list(plot_choices.values()):
    if plot_choice:
        plot_axs_values.append(idx)
        idx += 1
    else:
        plot_axs_values.append(None)
plot_axs = dict(zip(plot_axs_keys, plot_axs_values))

# Set up subplots
numplots = sum(plot_choices.values())
fig, ax = plt.subplots(nrows=numplots, ncols=1, figsize=(7, 7 * numplots))

# Plot!

if plot_choices["heatmap_ratio"]:
    heatmap_ablation_grid(
        ax[plot_axs["heatmap_ratio"]],
        exch_rate_dict,
        ratio_array,
        percent_saturation=True,
        vmin=axis_options["ratio_vmin"],
        vmax=axis_options["ratio_vmax"],
    )
    ax[plot_axs["heatmap_ratio"]].set_xlabel(grid_xlabel)
    ax[plot_axs["heatmap_ratio"]].set_ylabel(grid_ylabel)
    ax[plot_axs["heatmap_ratio"]].set_title("Ratio")

if plot_choices["heatmap_growthrate"]:
    heatmap_ablation_grid(
        ax[plot_axs["heatmap_growthrate"]],
        exch_rate_dict,
        growthrate_array,
        percent_saturation=True,
        cbar_label="growth rate",
    )
    ax[plot_axs["heatmap_growthrate"]].set_xlabel(grid_xlabel)
    ax[plot_axs["heatmap_growthrate"]].set_ylabel(grid_ylabel)
    ax[plot_axs["heatmap_growthrate"]].set_title("Growth rate")

if plot_choices["scatter_growthrate_ratio"]:
    ax[plot_axs["scatter_growthrate_ratio"]].scatter(growthrates, ratios)
    ax[plot_axs["scatter_growthrate_ratio"]].set_xlabel("Growth rate (/h)")
    ax[plot_axs["scatter_growthrate_ratio"]].set_ylabel("Ablation ratio")
    ax[plot_axs["scatter_growthrate_ratio"]].set_title("Growth rate vs ablation ratio")

if plot_choices["heatmap_gradient_c"]:
    heatmap_ablation_grid(
        ax[plot_axs["heatmap_gradient_c"]],
        exch_rate_dict,
        growthrate_gradient[0],
        percent_saturation=True,
        vmin=None,
        vmax=None,
        center=0,
        cmap="PiYG",
    )
    ax[plot_axs["heatmap_gradient_c"]].set_xlabel(grid_xlabel)
    ax[plot_axs["heatmap_gradient_c"]].set_ylabel(grid_ylabel)
    ax[plot_axs["heatmap_gradient_c"]].set_title(f"Gradient, {grid_xlabel_leader} axis")

if plot_choices["heatmap_gradient_n"]:
    heatmap_ablation_grid(
        ax[plot_axs["heatmap_gradient_n"]],
        exch_rate_dict,
        growthrate_gradient[1],
        percent_saturation=True,
        vmin=None,
        vmax=None,
        center=0,
        cmap="PiYG",
    )
    ax[plot_axs["heatmap_gradient_n"]].set_xlabel(grid_xlabel)
    ax[plot_axs["heatmap_gradient_n"]].set_ylabel(grid_ylabel)
    ax[plot_axs["heatmap_gradient_n"]].set_title(f"Gradient, {grid_ylabel_leader} axis")

if plot_choices["heatmap_gradient_compare"]:
    heatmap_ablation_grid(
        ax[plot_axs["heatmap_gradient_compare"]],
        exch_rate_dict,
        growthrate_gradient_greater,
        percent_saturation=True,
        vmin=None,
        vmax=None,
        center=0,
        cmap="PuOr",
    )
    ax[plot_axs["heatmap_gradient_compare"]].set_xlabel(grid_xlabel)
    ax[plot_axs["heatmap_gradient_compare"]].set_ylabel(grid_ylabel)
    ax[plot_axs["heatmap_gradient_compare"]].set_title(
        "1 = change in glucose axis has greater magnitude\n0 = change in ammonium axis has greater magnitude"
    )

if plot_choices["heatmap_ratio_whereone"]:
    heatmap_ablation_grid(
        ax[plot_axs["heatmap_ratio_whereone"]],
        exch_rate_dict,
        ratio_array_mask,
        percent_saturation=True,
        vmin=None,
        vmax=None,
        cmap="cividis",
    )
    ax[plot_axs["heatmap_ratio_whereone"]].set_xlabel(grid_xlabel)
    ax[plot_axs["heatmap_ratio_whereone"]].set_ylabel(grid_ylabel)
    ax[plot_axs["heatmap_ratio_whereone"]].set_title("0: ratio < 1; 1: ratio > 1")

if plot_choices["heatmap_carb_to_prot"]:
    heatmap_ablation_grid(
        ax[plot_axs["heatmap_carb_to_prot"]],
        exch_rate_dict,
        carb_to_prot_array,
        percent_saturation=True,
        vmin=None,
        vmax=None,
        cmap="Reds",
    )
    ax[plot_axs["heatmap_carb_to_prot"]].contour(np.rot90(ratio_array_mask))
    ax[plot_axs["heatmap_carb_to_prot"]].set_xlabel(grid_xlabel)
    ax[plot_axs["heatmap_carb_to_prot"]].set_ylabel(grid_ylabel)
    ax[plot_axs["heatmap_carb_to_prot"]].set_title("Carbohydrate:Protein ratio (times)")

if plot_choices["histogram_carb_to_prot"]:
    sns.histplot(
        data=carb_to_prot_df,
        x="carb_to_prot_ratio",
        hue="ratio_bool",
        element="step",
        binwidth=0.02,
        ax=ax[plot_axs["histogram_carb_to_prot"]],
    )
    ax[plot_axs["histogram_carb_to_prot"]].set_xlabel("Carbohydrate:Protein time ratio")
    ax[plot_axs["histogram_carb_to_prot"]].set_ylabel("Count")
    ax[plot_axs["histogram_carb_to_prot"]].get_legend().set_title("Ratio > 1")

pdf_filename = "../reports/" + filename + ".pdf"
with PdfPages(pdf_filename) as pdf:
    for fig in range(1, plt.gcf().number + 1):
        pdf.savefig(fig)
