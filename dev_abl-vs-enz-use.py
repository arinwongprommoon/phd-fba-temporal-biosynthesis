#!/usr/bin/env python3
#
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from yeast8model import Yeast8Model

glc_exch_rate = 16.89
wt_ec = Yeast8Model("./models/ecYeastGEM_batch_8-6-0.xml")
wt_ec.model.reactions.get_by_id("r_1714").bounds = (-glc_exch_rate, 0)
wt_ec.model.reactions.get_by_id("r_1714_REV").bounds = (0, glc_exch_rate)

wt_ec.ablation_result = wt_ec.ablate()

breakpoint()

ablation_fluxes = wt_ec.ablation_fluxes
ablation_fluxes_diff = ablation_fluxes.copy()
ablation_fluxes_diff.pop("original")
for biomass_component, fluxes in ablation_fluxes_diff.items():
    ablation_fluxes_diff[biomass_component] = (
        ablation_fluxes[biomass_component] - ablation_fluxes["original"]
    )

breakpoint()

fig, ax = plt.subplots(nrows=len(ablation_fluxes_diff), ncols=1)
for idx, (biomass_component, fluxes) in enumerate(ablation_fluxes_diff.items()):
    fluxes.plot.hist(ax=ax[idx])

breakpoint()
