#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.backends.backend_pdf import PdfPages
from src.data.biomasscomponent import biomass_component_list_orig
from src.gem.yeast8model import Yeast8Model
from src.viz.grid import piechart_ablation_grid

plot_options = {
    # "ec" or "y8"
    "model": "ec",
    # "glc" or "pyr"
    "carbon_source": "glc",
}

if plot_options["model"] == "ec":
    glc_exch_rate = 16.89
    wt = Yeast8Model("../data/gemfiles/ecYeastGEM_batch_8-6-0.xml")
    wt.model.reactions.get_by_id("r_1714").bounds = (-glc_exch_rate, 0)
    wt.model.reactions.get_by_id("r_1714_REV").bounds = (0, glc_exch_rate)

    exch_rate_dict = {
        "r_1714": np.linspace(0, 2 * 8.45, 8),  # glucose
        "r_2033": np.linspace(0, 2 * 4.27, 8),  # pyruvate
        "r_1654": np.linspace(0, 2 * 1.45, 8),  # ammonium
    }

elif plot_options["model"] == "y8":
    glucose_bounds = (-4.75, 0)  # gives a sensible growth rate for wt
    wt = Yeast8Model(
        "../data/gemfiles/yeast-GEM_8-6-0.xml", growth_id="r_2111", biomass_id="r_4041"
    )
    wt.biomass_component_list = biomass_component_list_orig
    wt.model.reactions.r_1714.bounds = glucose_bounds
    wt.add_media_components(["r_1992"])

    exch_rate_dict = {
        "r_1714": np.linspace(0, 2 * 4.75, 8),  # glucose
        "r_2033": np.linspace(0, 2 * 13.32, 8),  # pyruvate
        "r_1654": np.linspace(0, 2 * 2.88, 8),  # ammonium
    }

else:
    m = plot_options["model"]
    print(f"Invalid model {m}")


if plot_options["carbon_source"] == "glc":
    exch_rate_dict.pop("r_2033")
elif plot_options["carbon_source"] == "pyr":
    exch_rate_dict.pop("r_1714")

ablation_result_array = wt.ablation_grid(exch_rate_dict)

piechart_ablation_grid(
    exch_rate_dict,
    ablation_result_array,
    xlabel="Pyruvate exchange",
    ylabel="Ammonium exchange",
)

with PdfPages(f"piechart_plots.pdf") as pdf:
    for fig in range(1, plt.gcf().number + 1):
        pdf.savefig(fig)
