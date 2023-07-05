#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.backends.backend_pdf import PdfPages
from yeast8model import Yeast8Model, piechart_ablation_grid, biomass_component_list_orig

# ec
# glc_exch_rate = 16.89
# wt_ec = Yeast8Model("./models/ecYeastGEM_batch_8-6-0.xml")
# wt_ec.model.reactions.get_by_id("r_1714").bounds = (-glc_exch_rate, 0)
# wt_ec.model.reactions.get_by_id("r_1714_REV").bounds = (0, glc_exch_rate)

# non-ec
glucose_bounds = (-4.75, 0)  # gives a sensible growth rate for wt
wt_y8 = Yeast8Model(
    "./models/yeast-GEM_8-6-0.xml", growth_id="r_2111", biomass_id="r_4041"
)
wt_y8.biomass_component_list = biomass_component_list_orig
wt_y8.model.reactions.r_1714.bounds = glucose_bounds
wt_y8.add_media_components(["r_1992"])

# exch_rate_dict = {
#     "r_1714": np.linspace(0, 2 * 8.45, 8),  # glucose
#     "r_2033": np.linspace(0, 2 * 4.27, 8), # pyruvate
#     "r_1654": np.linspace(0, 2 * 1.45, 8),  # ammonium
#}
#ablation_result_array = wt_ec.ablation_grid(exch_rate_dict)

exch_rate_dict = {
#     "r_1714": np.linspace(0, 2 * 4.75, 8),  # glucose
    "r_2033": np.linspace(0, 2 * 13.32, 8), # pyruvate
    "r_1654": np.linspace(0, 2 * 2.88, 8),  # ammonium
}
ablation_result_array = wt_y8.ablation_grid(exch_rate_dict)

piechart_ablation_grid(
    exch_rate_dict,
    ablation_result_array,
    xlabel="Pyruvate exchange",
    ylabel="Ammonium exchange",
)

with PdfPages(f"piechart_plots.pdf") as pdf:
    for fig in range(1, plt.gcf().number + 1):
        pdf.savefig(fig)
