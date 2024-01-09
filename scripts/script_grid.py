#!/usr/bin/env python3
import operator
import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from src.calc.ablation import (
    vget_Tseq,
    vget_ablation_ratio,
    vget_custom_ablation_ratio,
    vget_cosine_carb_prot,
    vget_kendall_mean,
)
from src.calc.matrix import ArrayCollection
from src.viz.grid import heatmap_ablation_grid

model_options = {
    # "glc" or "pyr"
    "carbon_source": "glc",
}

plot_choices = {
    "heatmap_ratio": True,
    "heatmap_ratio_prot": False,
    "heatmap_ratio_prot_carb": False,
    "heatmap_ratio_prot_lipid": False,
    "heatmap_ratio_sus_compare": False,
    "heatmap_log2ratio": True,
    "heatmap_Tseq": True,
    "heatmap_gr": True,
    "heatmap_gr_gradient_c": False,
    "heatmap_gr_gradient_n": False,
    "heatmap_gr_gradient_compare": False,
    "heatmap_gr_sus_compare": False,
    "heatmap_carb": True,
    "heatmap_prot": True,
    "heatmap_carb_to_prot": True,
    "heatmap_cosine": False,
    "heatmap_kendall_mean": False,
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
grid_xlabel = f"{grid_xlabel_leader}\n (% growth rate saturation)"
grid_ylabel = f"{grid_ylabel_leader}\n (% growth rate saturation)"
# For quiver
X, Y = np.meshgrid(np.linspace(0, 31, 32), np.linspace(0, 31, 32))

# Load saved data
grid_filename = "ec_grid_" + model_options["carbon_source"] + "_amm"
grid_filepath = "../data/interim/" + grid_filename + ".pkl"
with open(grid_filepath, "rb") as handle:
    ablation_result_array = pickle.load(handle)

# usgfluxes_filename = "ec_usgfluxes_" + model_options["carbon_source"] + "_amm"
# usgfluxes_filepath = "../data/interim/" + usgfluxes_filename + ".pkl"
# with open(usgfluxes_filepath, "rb") as handle:
#     ablation_fluxes_array = pickle.load(handle)

# Compute data
ratio_array = vget_ablation_ratio(ablation_result_array)

ratio = ArrayCollection(ratio_array, x_axis, y_axis)
ratio_prot = ArrayCollection(
    vget_custom_ablation_ratio(ablation_result_array, ["protein"]), x_axis, y_axis
)
ratio_prot_carb = ArrayCollection(
    vget_custom_ablation_ratio(ablation_result_array, ["protein", "carbohydrate"]),
    x_axis,
    y_axis,
)
ratio_prot_lipid = ArrayCollection(
    vget_custom_ablation_ratio(ablation_result_array, ["protein", "lipid"]),
    x_axis,
    y_axis,
)

log2ratio = ArrayCollection(np.log2(ratio_array), x_axis, y_axis)

Tseq_array = vget_Tseq(ablation_result_array)
Tseq = ArrayCollection(Tseq_array, x_axis, y_axis)

gr = ArrayCollection(vget_gr(ablation_result_array), x_axis, y_axis)

carb = ArrayCollection(vget_carb(ablation_result_array), x_axis, y_axis)
prot = ArrayCollection(vget_prot(ablation_result_array), x_axis, y_axis)
carb_to_prot = ArrayCollection(carb.array / prot.array, x_axis, y_axis)

# kendall_mean = ArrayCollection(vget_kendall_mean(ablation_fluxes_array), x_axis, y_axis)

# Masks
ratio_array_mask = ratio.array > 1
# For growth rate gradients, non-zero as an 'epsilon' value to get sensible-
# looking contours.
gr_x_mask = gr.gradient.x > 0.01
gr_y_mask = gr.gradient.y > 0.01


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
    ratio_contour=True,
    isratio=False,
    gr_contour=False,
    quiver=False,
):
    """Convenience function to draw heatmaps with quivers

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
    ratio_contour : bool
       if true, draw ratio contour.  further options in 'isratio' parameter.
    isratio : bool
       if true, treats the input array as a ratio, and define contour based on
       where values are less than or greater than 1.  if false, draws contour
       based on the regular definition of ratio.
    gr_contour : bool
       if true, draw contours based on carbon- and nitrogen-limiting regions
       with respect to growth rate.
    quiver : bool
        if true, draw quiver based on susceptibility

    """
    data = operator.attrgetter(attribute)(acoll)
    heatmap_ablation_grid(
        ax,
        exch_rate_dict,
        data,
        percent_saturation=True,
        saturation_point=(saturation_carb, saturation_amm),
        saturation_grid=True,
        vmin=vmin,
        vmax=vmax,
        showticklabels=True,
        center=center,
        cmap=cmap,
        cbar_label=cbar_label,
    )
    if ratio_contour:
        if isratio:
            mask = data > 1
            ax.contour(np.rot90(mask), origin="lower", colors="k", linestyles="dotted")
        else:
            ax.contour(
                np.rot90(ratio_array_mask),
                origin="lower",
                colors="k",
                # linestyles="dotted",
            )
    if gr_contour:
        ax.contour(
            np.rot90(gr_x_mask), origin="lower", colors="C1", linestyles="dashed"
        )
        ax.contour(
            np.rot90(gr_y_mask), origin="lower", colors="C2", linestyles="dashed"
        )
    if quiver:
        ax.quiver(
            X,
            Y,
            acoll.sus_sp.y,
            -acoll.sus_sp.x,
            acoll.sus_sp.magnitudes,
            cmap="autumn",
        )
    ax.set_xlabel(grid_xlabel)
    ax.set_ylabel(grid_ylabel)
    ax.set_title(title)


# Plot!

# Set font size
# plt.rcParams.update({"font.size": 16})

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
        gr_contour=True,
        # quiver=True,
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
        isratio=True,
        quiver=True,
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
        isratio=True,
        quiver=True,
    )

if plot_choices["heatmap_ratio_prot_lipid"]:
    fig_heatmap_ratio_prot_lipid, ax_heatmap_ratio_prot_lipid = plt.subplots()
    riced_heatmap(
        ax_heatmap_ratio_prot_lipid,
        acoll=ratio_prot_lipid,
        cbar_label="Ratio",
        title="Ratio (from protein & lipid components only)",
        vmin=0.70,
        vmax=1.20,
        center=1,
        isratio=True,
        quiver=True,
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

if plot_choices["heatmap_log2ratio"]:
    fig_heatmap_log2ratio, ax_heatmap_log2ratio = plt.subplots()
    # This can safely re-use the contour computed on ratio because ratio > 1
    # is equivalent to log2ratio > 0.
    riced_heatmap(
        ax_heatmap_log2ratio,
        acoll=log2ratio,
        cbar_label=r"$\log_{2}(Ratio)$",
        title=r"$\log_{2}(Ratio)$",
        vmin=-0.52,
        vmax=+0.27,
        center=0,
        gr_contour=True,
        # quiver=True,
    )

if plot_choices["heatmap_Tseq"]:
    fig_heatmap_Tseq, ax_heatmap_Tseq = plt.subplots()
    riced_heatmap(
        ax_heatmap_Tseq,
        acoll=Tseq,
        cbar_label=r"T_{seq}",
        title=r"T_{seq}",
        # vmin=0.70,
        # vmax=1.20,
        # center=1,
        gr_contour=True,
        # quiver=True,
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
        ratio_contour=False,
        gr_contour=True,
        # quiver=True,
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
        ratio_contour=False,
        gr_contour=True,
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
        ratio_contour=False,
        gr_contour=True,
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
        ratio_contour=False,
        gr_contour=True,
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
        quiver=True,
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
        quiver=True,
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
        quiver=True,
    )

if plot_choices["heatmap_cosine"]:
    usgfluxes_filename = "ec_usgfluxes_" + model_options["carbon_source"] + "_amm"
    usgfluxes_filepath = "../data/interim/" + usgfluxes_filename + ".pkl"
    with open(usgfluxes_filepath, "rb") as handle:
        ablation_fluxes_array = pickle.load(handle)

    pdist = ArrayCollection(
        vget_cosine_carb_prot(ablation_fluxes_array), x_axis, y_axis
    )

    fig_heatmap_pdist, ax_heatmap_pdist = plt.subplots()
    riced_heatmap(
        ax_heatmap_pdist,
        acoll=pdist,
        cbar_label=r"Cosine distance",
        title="Cosine distance between carbohydrate\nand protein enzyme usage flux vectors",
        vmin=0,
        vmax=1,
        cmap="cividis_r",
        quiver=True,
    )

if plot_choices["heatmap_kendall_mean"]:
    fig_heatmap_kendall_mean, ax_heatmap_kendall_mean = plt.subplots()
    riced_heatmap(
        ax_heatmap_kendall_mean,
        acoll=kendall_mean,
        cbar_label=r"Mean Kendall's $\tau$ (b)",
        title="Mean correlation between parallel and each component",
        vmin=0.2,
        vmax=0.6,
        cmap="cividis",
        quiver=True,
    )

pdf_filename = "../reports/" + grid_filename + ".pdf"
with PdfPages(pdf_filename) as pdf:
    for fig in range(1, plt.gcf().number + 1):
        pdf.savefig(fig)
