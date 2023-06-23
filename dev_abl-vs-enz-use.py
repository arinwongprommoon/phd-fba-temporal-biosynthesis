#!/usr/bin/env python3
#
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from yeast8model import Yeast8Model

glc_exch_rate = 16.89
wt_ec = Yeast8Model("./models/ecYeastGEM_batch_8-6-0.xml")
wt_ec.model.reactions.get_by_id("r_1714").bounds = (-glc_exch_rate, 0)
wt_ec.model.reactions.get_by_id("r_1714_REV").bounds = (0, glc_exch_rate)

wt_ec.ablation_result = wt_ec.ablate()

ablation_fluxes = wt_ec.ablation_fluxes
ablation_fluxes_diff = ablation_fluxes.copy()
ablation_fluxes_diff.pop("original")
for biomass_component, fluxes in ablation_fluxes_diff.items():
    ablation_fluxes_diff[biomass_component] = (
        ablation_fluxes[biomass_component] - ablation_fluxes["original"]
    )
    print(f"{biomass_component}")
    print(f"min {1e5 * ablation_fluxes_diff[biomass_component].min()} * 1e-5")
    print(f"max {1e5 * ablation_fluxes_diff[biomass_component].max()} * 1e-5")

# suplots
# binrange=(-16e-5, ~90e-5) covers all diffs,
#   but using a smaller range to emphasise the interesting part
fig, ax = plt.subplots(nrows=len(ablation_fluxes_diff), ncols=1, sharex=True)
for idx, (biomass_component, fluxes) in enumerate(ablation_fluxes_diff.items()):
    sns.histplot(
        fluxes * 1e5, bins=100, binrange=(-16, +20), log_scale=(False, True), ax=ax[idx]
    )
    ax[idx].set_title(biomass_component)
    ax[idx].set_xlabel("")
    ax[idx].set_ylabel("")
# global labels
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
plt.grid(False)
plt.xlabel(r"Flux ($\times 10^{-5}$)")
plt.ylabel("Number of reactions")
plt.title("Changes in enzyme usage fluxes in biomass component ablation")

breakpoint()

plt.show()
