#!/usr/bin/env python3
import operator
import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from src.calc.ablation import vget_ablation_ratio, vget_custom_ablation_ratio
from src.calc.matrix import ArrayCollection
from src.viz.grid import heatmap_ablation_grid

model_options = {
    # "glc" or "pyr"
    "carbon_source": "glc",
}

plot_choices = {
    "heatmap_ratio": True,
    "heatmap_ratio_prot": True,
    "heatmap_ratio_prot_carb": True,
    "heatmap_ratio_sus_compare": True,
    "heatmap_gr": True,
    "heatmap_gr_gradient_c": False,
    "heatmap_gr_gradient_n": False,
    "heatmap_gr_gradient_compare": True,
    "heatmap_gr_sus_compare": True,
    "heatmap_carb": True,
    "heatmap_prot": True,
    "heatmap_carb_to_prot": True,
    "heatmap_eucl": True,
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
# For streamplot
X, Y = np.meshgrid(np.linspace(0, 31, 32), np.linspace(0, 31, 32))

# Load saved data
grid_filename = "ec_grid_" + model_options["carbon_source"] + "_amm"
grid_filepath = "../data/interim/" + grid_filename + ".pkl"
with open(grid_filepath, "rb") as handle:
    ablation_result_array = pickle.load(handle)

eucl_filename = "ec_eucl_" + model_options["carbon_source"] + "_amm"
eucl_filepath = "../data/interim/" + eucl_filename + ".pkl"
with open(eucl_filepath, "rb") as handle:
    eucl_array = pickle.load(handle)
# Convert dtype object to float, because of pickle
eucl_array = np.array(eucl_array, dtype=float)

# Compute data
ratio = ArrayCollection(vget_ablation_ratio(ablation_result_array), x_axis, y_axis)
ratio_prot = ArrayCollection(
    vget_custom_ablation_ratio(ablation_result_array, ["protein"]), x_axis, y_axis
)
ratio_prot_carb = ArrayCollection(
    vget_custom_ablation_ratio(ablation_result_array, ["protein", "carbohydrate"]),
    x_axis,
    y_axis,
)

gr = ArrayCollection(vget_gr(ablation_result_array), x_axis, y_axis)

carb = ArrayCollection(vget_carb(ablation_result_array), x_axis, y_axis)
prot = ArrayCollection(vget_prot(ablation_result_array), x_axis, y_axis)
carb_to_prot = ArrayCollection(carb.array / prot.array, x_axis, y_axis)

eucl = ArrayCollection(eucl_array, x_axis, y_axis)

# Mask
ratio_array_mask = ratio.array > 1


def riced_heatmap(
    ax,
    acoll,
    attribute="array",
    cbar_label=" ",
    title=" ",
    vmin=None,
    vmax=None,
    center=None,
    cmap="RdBu_r",
    streamplot=False,
):
    """Convenience function to draw heatmaps with streamplots

    Parameters
    ----------
    ax : matplotlib.pyplot Axes
        axes to draw on
    acoll : ArrayCollection
        array collection object
    attribute : string
        attribute of acoll to access to draw on heatmap. default "array"
    cbar_label : string
        colour bar label
    title : string
        title of plot
    vmin : float
        min value to show on heatmap
    vmax : float
        max value to show on heatmap
    center : float
        centre value for heatmap
    cmap : string
        matplotlib colour palette to use for colours
    streamplot : bool
        if true, draw streamplot based on susceptibility

    """
    heatmap_ablation_grid(
        ax,
        exch_rate_dict,
        operator.attrgetter(attribute)(acoll),
        percent_saturation=True,
        saturation_point=(saturation_carb, saturation_amm),
        saturation_grid=True,
        vmin=vmin,
        vmax=vmax,
        center=center,
        cmap=cmap,
        cbar_label=cbar_label,
    )
    ax.contour(np.rot90(ratio_array_mask), origin="lower")
    if streamplot:
        ax.streamplot(
            X,
            Y,
            acoll.sus_sp.y,
            acoll.sus_sp.x,
            color=acoll.sus_sp.magnitudes,
            arrowstyle="->",
            cmap="autumn",
        )
    ax.set_xlabel(grid_xlabel)
    ax.set_ylabel(grid_ylabel)
    ax.set_title(title)


# Plot!

if plot_choices["heatmap_ratio"]:
    fig_heatmap_ratio, ax_heatmap_ratio = plt.subplots()
    riced_heatmap(
        ax_heatmap_ratio,
        acoll=ratio,
        cbar_label="Ratio",
        title="Ratio",
        vmin=0.70,
        vmax=1.20,
        center=1,
        streamplot=True,
    )

if plot_choices["heatmap_ratio_prot"]:
    fig_heatmap_ratio_prot, ax_heatmap_ratio_prot = plt.subplots()
    riced_heatmap(
        ax_heatmap_ratio_prot,
        acoll=ratio_prot,
        cbar_label="Ratio",
        title="Ratio (from protein component only)",
        vmin=0.70,
        vmax=1.20,
        center=1,
        streamplot=True,
    )

if plot_choices["heatmap_ratio_prot_carb"]:
    fig_heatmap_ratio_prot_carb, ax_heatmap_ratio_prot_carb = plt.subplots()
    riced_heatmap(
        ax_heatmap_ratio_prot_carb,
        acoll=ratio_prot_carb,
        cbar_label="Ratio",
        title="Ratio (from protein & carbohydrate components only)",
        vmin=0.70,
        vmax=1.20,
        center=1,
        streamplot=True,
    )

if plot_choices["heatmap_ratio_sus_compare"]:
    fig_heatmap_ratio_sus_compare, ax_heatmap_ratio_sus_compare = plt.subplots()
    riced_heatmap(
        ax_heatmap_ratio_sus_compare,
        acoll=ratio,
        attribute="sus.greater",
        cbar_label="Susceptibility difference",
        title=f"Differences in magnitude of susceptibility of ratio,\n{grid_xlabel_leader} -- {grid_ylabel_leader}",
        center=0,
        cmap="PuOr",
    )


if plot_choices["heatmap_gr"]:
    fig_heatmap_gr, ax_heatmap_gr = plt.subplots()
    riced_heatmap(
        ax_heatmap_gr,
        acoll=gr,
        cbar_label="Growth rate",
        title="Growth rate",
        vmin=0,
        vmax=0.40,
        cmap="cividis",
        streamplot=True,
    )

if plot_choices["heatmap_gr_gradient_c"]:
    fig_heatmap_gr_gradient_c, ax_heatmap_gradient_c = plt.subplots()
    riced_heatmap(
        ax_heatmap_gradient_c,
        acoll=gr,
        attribute="gradient.x",
        cbar_label="Gradient",
        title=f"Gradient of growth rate,\nalong {grid_xlabel_leader} axis",
        cmap="PiYG",
    )

if plot_choices["heatmap_gr_gradient_n"]:
    fig_heatmap_gr_gradient_n, ax_heatmap_gradient_n = plt.subplots()
    riced_heatmap(
        ax_heatmap_gradient_n,
        acoll=gr,
        attribute="gradient.y",
        cbar_label="Gradient",
        title=f"Gradient of growth rate,\nalong {grid_ylabel_leader} axis",
        cmap="PiYG",
    )

if plot_choices["heatmap_gr_gradient_compare"]:
    fig_heatmap_gr_gradient_compare, ax_heatmap_gr_gradient_compare = plt.subplots()
    riced_heatmap(
        ax_heatmap_gr_gradient_compare,
        acoll=gr,
        attribute="gradient.greater",
        cbar_label="Gradient difference",
        title=f"Differences in magnitude of gradient,\n{grid_xlabel_leader} -- {grid_ylabel_leader}",
        center=0,
        cmap="PuOr",
    )

if plot_choices["heatmap_gr_sus_compare"]:
    fig_heatmap_gr_sus_compare, ax_heatmap_gr_sus_compare = plt.subplots()
    riced_heatmap(
        ax_heatmap_gr_sus_compare,
        acoll=gr,
        attribute="sus.greater",
        cbar_label="Susceptibility difference",
        title=f"Differences in magnitude of susceptibility,\n{grid_xlabel_leader} -- {grid_ylabel_leader}",
        cmap="PuOr",
    )

if plot_choices["heatmap_carb"]:
    fig_heatmap_carb, ax_heatmap_carb = plt.subplots()
    riced_heatmap(
        ax_heatmap_carb,
        acoll=carb,
        cbar_label="Time (hours)",
        title="Predicted carbohydrate synthesis time",
        vmin=0,
        vmax=3,
        cmap="Reds",
        streamplot=True,
    )

if plot_choices["heatmap_prot"]:
    fig_heatmap_prot, ax_heatmap_prot = plt.subplots()
    riced_heatmap(
        ax_heatmap_prot,
        acoll=prot,
        cbar_label="Time (hours)",
        title="Predicted protein synthesis time",
        vmin=0,
        vmax=10,
        cmap="Blues",
        streamplot=True,
    )

if plot_choices["heatmap_carb_to_prot"]:
    fig_heatmap_carb_to_prot, ax_heatmap_carb_to_prot = plt.subplots()
    riced_heatmap(
        ax_heatmap_carb_to_prot,
        acoll=carb_to_prot,
        cbar_label="Ratio",
        title="Ratio of carbohydrate synthesis time\nto protein synthesis time",
        vmin=0,
        vmax=0.5,
        cmap="Purples",
        streamplot=True,
    )

if plot_choices["heatmap_eucl"]:
    fig_heatmap_eucl, ax_heatmap_eucl = plt.subplots()
    riced_heatmap(
        ax_heatmap_eucl,
        acoll=eucl,
        cbar_label="Distance",
        title="Euclidean distance between enzyme usage flux vectors:\nprioritising protein vs carbohydrate",
        vmin=0,
        vmax=0.0020,
        cmap="magma_r",
    )

pdf_filename = "../reports/" + grid_filename + ".pdf"
with PdfPages(pdf_filename) as pdf:
    for fig in range(1, plt.gcf().number + 1):
        pdf.savefig(fig)
