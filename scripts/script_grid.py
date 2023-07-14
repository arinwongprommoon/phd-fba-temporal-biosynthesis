#!/usr/bin/env python3
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

from src.calc.ablation import vget_ablation_ratio
from src.calc.matrix import get_susceptibility
from src.viz.grid import heatmap_ablation_grid

model_options = {
    # "glc" or "pyr"
    "carbon_source": "glc",
}

plot_choices = {
    "heatmap_ratio": True,
    "heatmap_ratio_sus_compare": True,
    "heatmap_gr": True,
    "heatmap_gr_gradient_c": False,
    "heatmap_gr_gradient_n": False,
    "heatmap_gr_gradient_compare": True,
    "heatmap_gr_sus_compare": True,
    "heatmap_ratio_whereone": False,
    "heatmap_carb": True,
    "heatmap_prot": True,
    "heatmap_carb_to_prot": True,
}


@np.vectorize
def vget_gr(x):
    return x.ablated_flux[0]


@np.vectorize
def vget_carb(x):
    carb = x.ablated_est_time[3]
    return carb


@np.vectorize
def vget_prot(x):
    prot = x.ablated_est_time[2]
    return prot


saturation_glc = 8.6869
saturation_pyr = 4.4444

exch_rate_dict = {}

if model_options["carbon_source"] == "glc":
    # build exch_rate_dict
    saturation_amm = 1.4848
    exch_rate_dict["r_1714"] = np.linspace(0, 2 * saturation_glc, 32)
    # plot options
    x_axis = exch_rate_dict["r_1714"]
    saturation_carb = saturation_glc
    grid_xlabel_leader = "Glucose exchange"
elif model_options["carbon_source"] == "pyr":
    # build exch_rate_dict
    saturation_amm = 1.0
    exch_rate_dict["r_2033"] = np.linspace(0, 2 * saturation_pyr, 32)
    # plot options
    x_axis = exch_rate_dict["r_2033"]
    saturation_carb = saturation_pyr
    grid_xlabel_leader = "Pyruvate exchange"

exch_rate_dict["r_1654"] = np.linspace(0, 2 * saturation_amm, 32)
y_axis = exch_rate_dict["r_1654"]
grid_ylabel_leader = "Ammonium exchange"

# Set up axes parameters
grid_xlabel = f"{grid_xlabel_leader} (% saturation)"
grid_ylabel = f"{grid_ylabel_leader} (% saturation)"


# Load saved data
filename = "ec_grid_" + model_options["carbon_source"] + "_amm"
filepath = "../data/interim/" + filename + ".pkl"
with open(filepath, "rb") as handle:
    ablation_result_array = pickle.load(handle)


# Compute data
X, Y = np.meshgrid(np.linspace(0, 31, 32), np.linspace(0, 31, 32))


class ArrayCollection:
    def __init__(self, raw_array, x_axis, y_axis):
        # Store raw input
        self.raw = raw_array

        # Main use
        self.array = self.raw
        # Replace pixels that correspond to exch rate 0 with NaNs
        # These are prone to weird values
        self.array[0, :] = np.nan
        self.array[:, 0] = np.nan

        # Susceptibility
        _sus = get_susceptibility(self.array, x_axis, y_axis)
        self.sus = GradientCollection(*_sus)

        # Gradient
        _gradient = np.gradient(self.array)
        self.gradient = GradientCollection(*_gradient)

        # For streamplots
        # rot90 and flipping values to get orientation & arrow right because
        # matplotlib and seaborn use different axes directions
        # Only sus for now -- will extend to gradient when needed
        _sus_sp = get_susceptibility(np.rot90(self.array), x_axis, y_axis[::-1])
        self.sus_sp = StreamplotInputs(*_sus_sp)


class GradientCollection:
    def __init__(self, x_gradient_array, y_gradient_array):
        # Store raw input
        self.x = x_gradient_array
        self.y = y_gradient_array

        # Magnitude
        self.magnitudes = np.sqrt(self.x**2, self.y**2)

        # Magnitude of x subtracted by magnitude of y
        # Useful for seeing which area is x- or y-limiting
        self.greater = np.abs(self.x) - np.abs(self.y)


class StreamplotInputs:
    def __init__(self, x_gradient_array, y_gradient_array):
        # Store raw input
        self.x = x_gradient_array
        self.y = y_gradient_array

        # Flipping to play well with streamplot
        self.y = -self.y

        # Magnitude
        self.magnitudes = np.sqrt(self.x**2, self.y**2)


ratio = ArrayCollection(vget_ablation_ratio(ablation_result_array), x_axis, y_axis)
gr = ArrayCollection(vget_gr(ablation_result_array), x_axis, y_axis)
carb = ArrayCollection(vget_carb(ablation_result_array), x_axis, y_axis)
prot = ArrayCollection(vget_prot(ablation_result_array), x_axis, y_axis)
carb_to_prot = ArrayCollection(carb.array / prot.array, x_axis, y_axis)

ratio_array_mask = ratio.array > 1


# Plot!

if plot_choices["heatmap_ratio"]:
    fig_heatmap_ratio, ax_heatmap_ratio = plt.subplots()
    heatmap_ablation_grid(
        ax_heatmap_ratio,
        exch_rate_dict,
        ratio.array,
        percent_saturation=True,
        saturation_point=(saturation_carb, saturation_amm),
        saturation_grid=True,
        vmin=0.70,
        vmax=1.20,
        center=1,
        cbar_label="Ratio",
    )
    ax_heatmap_ratio.contour(np.rot90(ratio_array_mask), origin="lower")
    ax_heatmap_ratio.streamplot(
        X,
        Y,
        ratio.sus_sp.y,
        ratio.sus_sp.x,
        color=ratio.sus_sp.magnitudes,
        arrowstyle="->",
        cmap="autumn",
    )
    ax_heatmap_ratio.set_xlabel(grid_xlabel)
    ax_heatmap_ratio.set_ylabel(grid_ylabel)
    ax_heatmap_ratio.set_title("Ratio")

if plot_choices["heatmap_ratio_sus_compare"]:
    fig_heatmap_ratio_sus_compare, ax_heatmap_ratio_sus_compare = plt.subplots()
    heatmap_ablation_grid(
        ax_heatmap_ratio_sus_compare,
        exch_rate_dict,
        ratio_sus_greater,
        percent_saturation=True,
        saturation_point=(saturation_carb, saturation_amm),
        saturation_grid=True,
        vmin=None,
        vmax=None,
        center=0,
        cmap="PuOr",
        cbar_label="Susceptibility difference",
    )
    ax_heatmap_ratio_sus_compare.contour(np.rot90(ratio_array_mask), origin="lower")
    ax_heatmap_ratio_sus_compare.set_xlabel(grid_xlabel)
    ax_heatmap_ratio_sus_compare.set_ylabel(grid_ylabel)
    ax_heatmap_ratio_sus_compare.set_title(
        f"Differences in magnitude of susceptibility of ratio,\n{grid_xlabel_leader} -- {grid_ylabel_leader}"
    )

if plot_choices["heatmap_gr"]:
    fig_heatmap_gr, ax_heatmap_gr = plt.subplots()
    heatmap_ablation_grid(
        ax_heatmap_gr,
        exch_rate_dict,
        gr_array,
        percent_saturation=True,
        saturation_point=(saturation_carb, saturation_amm),
        saturation_grid=True,
        vmin=0,
        vmax=0.40,
        cmap="cividis",
        cbar_label="Growth rate",
    )
    ax_heatmap_gr.contour(np.rot90(ratio_array_mask), origin="lower")
    ax_heatmap_gr.streamplot(
        X,
        Y,
        gr_sus_rot90[1],
        gr_sus_rot90[0],
        color=gr_sus_magnitudes_rot90,
        arrowstyle="->",
        cmap="autumn",
    )
    ax_heatmap_gr.set_xlabel(grid_xlabel)
    ax_heatmap_gr.set_ylabel(grid_ylabel)
    ax_heatmap_gr.set_title("Growth rate")

if plot_choices["heatmap_gr_gradient_c"]:
    fig_heatmap_gr_gradient_c, ax_heatmap_gr_gradient_c = plt.subplots()
    heatmap_ablation_grid(
        ax_heatmap_gr_gradient_c,
        exch_rate_dict,
        gr_gradient[0],
        percent_saturation=True,
        saturation_point=(saturation_carb, saturation_amm),
        saturation_grid=True,
        vmin=None,
        vmax=None,
        center=0,
        cmap="PiYG",
        cbar_label="Gradient",
    )
    ax_heatmap_gr_gradient_c.contour(np.rot90(ratio_array_mask), origin="lower")
    ax_heatmap_gr_gradient_c.set_xlabel(grid_xlabel)
    ax_heatmap_gr_gradient_c.set_ylabel(grid_ylabel)
    ax_heatmap_gr_gradient_c.set_title(
        f"Gradient of growth rate,\nalong {grid_xlabel_leader} axis"
    )

if plot_choices["heatmap_gr_gradient_n"]:
    fig_heatmap_gr_gradient_n, ax_heatmap_gr_gradient_n = plt.subplots()
    heatmap_ablation_grid(
        ax_heatmap_gr_gradient_n,
        exch_rate_dict,
        gr_gradient[1],
        percent_saturation=True,
        saturation_point=(saturation_carb, saturation_amm),
        saturation_grid=True,
        vmin=None,
        vmax=None,
        center=0,
        cmap="PiYG",
        cbar_label="Gradient",
    )
    ax_heatmap_gr_gradient_n.contour(np.rot90(ratio_array_mask), origin="lower")
    ax_heatmap_gr_gradient_n.set_xlabel(grid_xlabel)
    ax_heatmap_gr_gradient_n.set_ylabel(grid_ylabel)
    ax_heatmap_gr_gradient_n.set_title(
        f"Gradient of growth rate,\nalong {grid_ylabel_leader} axis"
    )

if plot_choices["heatmap_gr_gradient_compare"]:
    fig_heatmap_gr_gradient_compare, ax_heatmap_gr_gradient_compare = plt.subplots()
    heatmap_ablation_grid(
        ax_heatmap_gr_gradient_compare,
        exch_rate_dict,
        gr_gradient_greater,
        percent_saturation=True,
        saturation_point=(saturation_carb, saturation_amm),
        saturation_grid=True,
        vmin=None,
        vmax=None,
        center=0,
        cmap="PuOr",
        cbar_label="Gradient difference",
    )
    ax_heatmap_gr_gradient_compare.contour(np.rot90(ratio_array_mask), origin="lower")
    ax_heatmap_gr_gradient_compare.set_xlabel(grid_xlabel)
    ax_heatmap_gr_gradient_compare.set_ylabel(grid_ylabel)
    ax_heatmap_gr_gradient_compare.set_title(
        f"Differences in magnitude of gradient,\n{grid_xlabel_leader} -- {grid_ylabel_leader}"
    )

if plot_choices["heatmap_gr_sus_compare"]:
    fig_heatmap_gr_sus_compare, ax_heatmap_gr_sus_compare = plt.subplots()
    heatmap_ablation_grid(
        ax_heatmap_gr_sus_compare,
        exch_rate_dict,
        gr_sus_greater,
        percent_saturation=True,
        saturation_point=(saturation_carb, saturation_amm),
        saturation_grid=True,
        vmin=None,
        vmax=None,
        center=0,
        cmap="PuOr",
        cbar_label="Susceptibility difference",
    )
    ax_heatmap_gr_sus_compare.contour(np.rot90(ratio_array_mask), origin="lower")
    ax_heatmap_gr_sus_compare.set_xlabel(grid_xlabel)
    ax_heatmap_gr_sus_compare.set_ylabel(grid_ylabel)
    ax_heatmap_gr_sus_compare.set_title(
        f"Differences in magnitude of susceptibility of growth rate,\n{grid_xlabel_leader} -- {grid_ylabel_leader}"
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
    ax_heatmap_carb.contour(np.rot90(ratio_array_mask), origin="lower")
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
    ax_heatmap_prot.contour(np.rot90(ratio_array_mask), origin="lower")
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
        vmin=0,
        vmax=0.5,
        cmap="Purples",
        cbar_label="Ratio",
    )
    ax_heatmap_carb_to_prot.contour(np.rot90(ratio_array_mask), origin="lower")
    ax_heatmap_carb_to_prot.set_xlabel(grid_xlabel)
    ax_heatmap_carb_to_prot.set_ylabel(grid_ylabel)
    ax_heatmap_carb_to_prot.set_title(
        "Ratio of carbohydrate synthesis time\nto protein synthesis time"
    )

pdf_filename = "../reports/" + filename + ".pdf"
with PdfPages(pdf_filename) as pdf:
    for fig in range(1, plt.gcf().number + 1):
        pdf.savefig(fig)
