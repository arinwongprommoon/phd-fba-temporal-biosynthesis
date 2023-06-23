#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.backends.backend_pdf import PdfPages
from yeast8model import (
    Yeast8Model,
    heatmap_ablation_grid,
    vget_ablation_ratio,
    biomass_component_list_orig,
)

# CHOOSE MODEL
if True:
    glc_exch_rate = 16.89
    wt_ec = Yeast8Model("./models/ecYeastGEM_batch_8-6-0.xml")
    wt_ec.model.reactions.get_by_id("r_1714").bounds = (-glc_exch_rate, 0)
    wt_ec.model.reactions.get_by_id("r_1714_REV").bounds = (0, glc_exch_rate)

if False:
    glucose_bounds = (-4.75, 0)  # gives a sensible growth rate for wt
    wt_y8 = Yeast8Model(
        "./models/yeast-GEM_8-6-0.xml", growth_id="r_2111", biomass_id="r_4041"
    )
    wt_y8.biomass_component_list = biomass_component_list_orig
    wt_y8.model.reactions.r_1714.bounds = glucose_bounds
    wt_y8.add_media_components(["r_1992"])

# PARAMETERS
# Step down, because 'phantom' values or copying issues (that I don't want to fix)
fractions = np.linspace(1, 0, num=6)

exch_rate_dict = {
    "r_1714": np.linspace(0, 2 * 8.45, 16),  # glucose
    "r_1654": np.linspace(0, 2 * 1.45, 16),  # ammonium
}

# WORK
sol = wt_ec.optimize()
orig_flux_sum = sol.fluxes.abs().sum()

ablation_result_array_list = []

for fraction in fractions:
    print(f'CONSTRAIN FLUXES fraction {fraction}')
    ub = fraction * orig_flux_sum
    wt_ec.set_flux_constraint(upper_bound=ub)
    sol = wt_ec.optimize()
    ablation_result_array = wt_ec.ablation_grid(exch_rate_dict)
    ablation_result_array_list.append(ablation_result_array)

# PLOT
grid_xlabel_leader = "Glucose exchange"
grid_ylabel_leader = "Ammonium exchange"

xmax = np.max(list(exch_rate_dict.values())[0])
ymax = np.max(list(exch_rate_dict.values())[0])
grid_xlabel = f"{grid_xlabel_leader} (% max = {xmax:.2f})"
grid_ylabel = f"{grid_ylabel_leader} (% max = {ymax:.2f})"

nsubfig = len(fractions)
fig, ax = plt.subplots(nrows=nsubfig, ncols=1, figsize=(7, 6 * nsubfig))

for idx, fraction in enumerate(fractions):
    ratio_array = vget_ablation_ratio(ablation_result_array_list[idx])
    heatmap_ablation_grid(
        ax[idx],
        exch_rate_dict,
        ratio_array,
        percent_saturation=True,
        vmin=0.5,
        vmax=1.5,
    )
    ax[idx].set_xlabel(grid_xlabel)
    ax[idx].set_ylabel(grid_ylabel)
    ax[idx].set_title(f"Constraint: {fraction:.3f} of max sum of fluxes")

with PdfPages(f"constrain_fluxes_plots.pdf") as pdf:
    for fig in range(1, plt.gcf().number + 1):
        pdf.savefig(fig)
