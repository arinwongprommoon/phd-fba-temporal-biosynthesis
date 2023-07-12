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
    "scatter_growthrate_ratio": False,
    "heatmap_gradient_c": False,
    "heatmap_gradient_n": False,
    "heatmap_gradient_compare": True,
    "heatmap_ratio_whereone": False,
    "heatmap_carb": True,
    "heatmap_prot": True,
    "heatmap_carb_to_prot": True,
    "histogram_carb_to_prot": False,
}


@np.vectorize
def vget_growthrate(x):
    return x.ablated_flux[0]


@np.vectorize
def vget_carb(x):
    carb = x.ablated_est_time[3]
    return carb


@np.vectorize
def vget_prot(x):
    prot = x.ablated_est_time[2]
    return prot


if model_options["model"] == "ec":
    saturation_glc = 8.6869
    saturation_pyr = 4.4444
    saturation_amm = 1.4848
    exch_rate_dict = {
        "r_1714": np.linspace(0, 2 * saturation_glc, 32),  # glucose
        "r_2033": np.linspace(0, 2 * saturation_pyr, 32),  # pyruvate
        "r_1654": np.linspace(0, 2 * saturation_amm, 32),  # ammonium
    }
elif model_options["model"] == "y8":
    saturation_glc = 4.75
    saturation_pyr = 13.32
    saturation_amm = 2.88
    exch_rate_dict = {
        "r_1714": np.linspace(0, 2 * saturation_glc, 32),  # glucose
        "r_2033": np.linspace(0, 2 * saturation_pyr, 32),  # pyruvate
        "r_1654": np.linspace(0, 2 * saturation_amm, 32),  # ammonium
    }
else:
    m = model_options["model"]
    print(f"Invalid model {m}")

if model_options["carbon_source"] == "glc":
    exch_rate_dict.pop("r_2033")
    axis_options["grid_xlabel_leader"] = "Glucose exchange"
    saturation_carb = saturation_glc
elif model_options["carbon_source"] == "pyr":
    exch_rate_dict.pop("r_1714")
    # bodge
    saturation_amm = 1.0
    exch_rate_dict["r_1654"] = np.linspace(0, 2 * saturation_amm, 32)
    axis_options["grid_xlabel_leader"] = "Pyruvate exchange"
    saturation_carb = saturation_pyr

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
carb_array = vget_carb(ablation_result_array)
prot_array = vget_prot(ablation_result_array)
carb_to_prot_array = carb_array / prot_array

# Set up axes parameters
grid_xlabel_leader = axis_options["grid_xlabel_leader"]
grid_ylabel_leader = axis_options["grid_ylabel_leader"]
grid_xlabel = f"{grid_xlabel_leader} (% saturation)"
grid_ylabel = f"{grid_ylabel_leader} (% saturation)"

# Plot!

if plot_choices["heatmap_ratio"]:
    fig_heatmap_ratio, ax_heatmap_ratio = plt.subplots()
    heatmap_ablation_grid(
        ax_heatmap_ratio,
        exch_rate_dict,
        ratio_array,
        percent_saturation=True,
        saturation_point=(saturation_carb, saturation_amm),
        saturation_grid=True,
        vmin=axis_options["ratio_vmin"],
        vmax=axis_options["ratio_vmax"],
        cbar_label="Ratio",
    )
    ax_heatmap_ratio.contour(np.rot90(ratio_array_mask))
    ax_heatmap_ratio.set_xlabel(grid_xlabel)
    ax_heatmap_ratio.set_ylabel(grid_ylabel)
    ax_heatmap_ratio.set_title("Ratio")

if plot_choices["heatmap_growthrate"]:
    fig_heatmap_growthrate, ax_heatmap_growthrate = plt.subplots()
    heatmap_ablation_grid(
        ax_heatmap_growthrate,
        exch_rate_dict,
        growthrate_array,
        percent_saturation=True,
        saturation_point=(saturation_carb, saturation_amm),
        saturation_grid=True,
        cbar_label="Growth rate",
    )
    ax_heatmap_growthrate.contour(np.rot90(ratio_array_mask))
    ax_heatmap_growthrate.set_xlabel(grid_xlabel)
    ax_heatmap_growthrate.set_ylabel(grid_ylabel)
    ax_heatmap_growthrate.set_title("Growth rate")

if plot_choices["scatter_growthrate_ratio"]:
    ratios = ratio_array[1:, 1:].ravel()
    growthrates = growthrate_array[1:, 1:].ravel()
    fig_heatmap_scatter_growthrate_ratio, ax_scatter_growthrate_ratio = plt.subplots()
    ax_scatter_growthrate_ratio.scatter(growthrates, ratios)
    ax_scatter_growthrate_ratio.set_xlabel(r"Growth rate ($h^{-1}$)")
    ax_scatter_growthrate_ratio.set_ylabel("Ablation ratio")
    ax_scatter_growthrate_ratio.set_title("Growth rate vs ablation ratio")

if plot_choices["heatmap_gradient_c"]:
    fig_heatmap_gradient_c, ax_heatmap_gradient_c = plt.subplots()
    heatmap_ablation_grid(
        ax_heatmap_gradient_c,
        exch_rate_dict,
        growthrate_gradient[0],
        percent_saturation=True,
        saturation_point=(saturation_carb, saturation_amm),
        saturation_grid=True,
        vmin=None,
        vmax=None,
        center=0,
        cmap="PiYG",
        cbar_label="Gradient",
    )
    ax_heatmap_gradient_c.contour(np.rot90(ratio_array_mask))
    ax_heatmap_gradient_c.set_xlabel(grid_xlabel)
    ax_heatmap_gradient_c.set_ylabel(grid_ylabel)
    ax_heatmap_gradient_c.set_title(
        f"Gradient of growth rate,\nalong {grid_xlabel_leader} axis"
    )

if plot_choices["heatmap_gradient_n"]:
    fig_heatmap_gradient_n, ax_heatmap_gradient_n = plt.subplots()
    heatmap_ablation_grid(
        ax_heatmap_gradient_n,
        exch_rate_dict,
        growthrate_gradient[1],
        percent_saturation=True,
        saturation_point=(saturation_carb, saturation_amm),
        saturation_grid=True,
        vmin=None,
        vmax=None,
        center=0,
        cmap="PiYG",
        cbar_label="Gradient",
    )
    ax_heatmap_gradient_n.contour(np.rot90(ratio_array_mask))
    ax_heatmap_gradient_n.set_xlabel(grid_xlabel)
    ax_heatmap_gradient_n.set_ylabel(grid_ylabel)
    ax_heatmap_gradient_n.set_title(
        f"Gradient of growth rate,\nalong {grid_ylabel_leader} axis"
    )

if plot_choices["heatmap_gradient_compare"]:
    fig_heatmap_gradient_compare, ax_heatmap_gradient_compare = plt.subplots()
    heatmap_ablation_grid(
        ax_heatmap_gradient_compare,
        exch_rate_dict,
        growthrate_gradient_greater,
        percent_saturation=True,
        saturation_point=(saturation_carb, saturation_amm),
        saturation_grid=True,
        vmin=None,
        vmax=None,
        center=0,
        cmap="PuOr",
        cbar_label="Gradient difference",
    )
    ax_heatmap_gradient_compare.contour(np.rot90(ratio_array_mask))
    ax_heatmap_gradient_compare.set_xlabel(grid_xlabel)
    ax_heatmap_gradient_compare.set_ylabel(grid_ylabel)
    ax_heatmap_gradient_compare.set_title(
        f"Differences in magnitude of gradient,\n{grid_xlabel_leader} -- {grid_ylabel_leader}"
    )

if plot_choices["heatmap_ratio_whereone"]:
    fig_heatmap_ratio_whereone, ax_heatmap_ratio_whereone = plt.subplots()
    heatmap_ablation_grid(
        ax_heatmap_ratio_whereone,
        exch_rate_dict,
        ratio_array_mask,
        percent_saturation=True,
        saturation_point=(saturation_carb, saturation_amm),
        saturation_grid=True,
        vmin=None,
        vmax=None,
        cmap="cividis",
        cbar_label=" ",
    )
    ax_heatmap_ratio_whereone.set_xlabel(grid_xlabel)
    ax_heatmap_ratio_whereone.set_ylabel(grid_ylabel)
    ax_heatmap_ratio_whereone.set_title(r"Conditions in which $r > 1$")

if plot_choices["heatmap_carb"]:
    fig_heatmap_carb, ax_heatmap_carb = plt.subplots()
    heatmap_ablation_grid(
        ax_heatmap_carb,
        exch_rate_dict,
        carb_array,
        percent_saturation=True,
        saturation_point=(saturation_carb, saturation_amm),
        saturation_grid=True,
        vmin=0,
        vmax=3,
        cmap="Reds",
        cbar_label="Time (hours)",
    )
    ax_heatmap_carb.contour(np.rot90(ratio_array_mask))
    ax_heatmap_carb.set_xlabel(grid_xlabel)
    ax_heatmap_carb.set_ylabel(grid_ylabel)
    ax_heatmap_carb.set_title("Predicted carbohydrate synthesis time")


if plot_choices["heatmap_prot"]:
    fig_heatmap_prot, ax_heatmap_prot = plt.subplots()
    heatmap_ablation_grid(
        ax_heatmap_prot,
        exch_rate_dict,
        prot_array,
        percent_saturation=True,
        saturation_point=(saturation_carb, saturation_amm),
        saturation_grid=True,
        vmin=0,
        vmax=10,
        cmap="Blues",
        cbar_label="Time (hours)",
    )
    ax_heatmap_prot.contour(np.rot90(ratio_array_mask))
    ax_heatmap_prot.set_xlabel(grid_xlabel)
    ax_heatmap_prot.set_ylabel(grid_ylabel)
    ax_heatmap_prot.set_title("Predicted protein synthesis time")


if plot_choices["heatmap_carb_to_prot"]:
    fig_heatmap_carb_to_prot, ax_heatmap_carb_to_prot = plt.subplots()
    heatmap_ablation_grid(
        ax_heatmap_carb_to_prot,
        exch_rate_dict,
        carb_to_prot_array,
        percent_saturation=True,
        saturation_point=(saturation_carb, saturation_amm),
        saturation_grid=True,
        vmin=None,
        vmax=None,
        cmap="Purples",
        cbar_label="Ratio",
    )
    ax_heatmap_carb_to_prot.contour(np.rot90(ratio_array_mask))
    ax_heatmap_carb_to_prot.set_xlabel(grid_xlabel)
    ax_heatmap_carb_to_prot.set_ylabel(grid_ylabel)
    ax_heatmap_carb_to_prot.set_title(
        "Ratio of carbohydrate synthesis time\nto protein synthesis time"
    )

if plot_choices["histogram_carb_to_prot"]:
    carb_to_prot_ratios = carb_to_prot_array[1:, 1:].ravel()
    ratio_bools = ratio_array_mask[1:, 1:].ravel()
    carb_to_prot_df = pd.DataFrame(
        {
            "carb_to_prot_ratio": carb_to_prot_ratios,
            "ratio_bool": ratio_bools,
        }
    )
    fig_histogram_carb_to_prot, ax_histogram_carb_to_prot = plt.subplots()
    sns.histplot(
        data=carb_to_prot_df,
        x="carb_to_prot_ratio",
        hue="ratio_bool",
        element="step",
        binwidth=0.02,
        ax=ax_histogram_carb_to_prot,
    )
    ax_histogram_carb_to_prot.set_xlabel("Carbohydrate:Protein time ratio")
    ax_histogram_carb_to_prot.set_ylabel("Count")
    ax_histogram_carb_to_prot.get_legend().set_title("Ratio > 1")

pdf_filename = "../reports/" + filename + ".pdf"
with PdfPages(pdf_filename) as pdf:
    for fig in range(1, plt.gcf().number + 1):
        pdf.savefig(fig)
