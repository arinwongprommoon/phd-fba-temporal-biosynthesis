#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cobra

from yeast8model import Yeast8Model
from yeast8model import heatmap_ablation_grid

glc_exch_rate = 16.89
wt_ec = Yeast8Model("./models/ecYeastGEM_batch_8-6-0.xml")
wt_ec.model.reactions.get_by_id("r_1714").bounds = (-glc_exch_rate, 0)
wt_ec.model.reactions.get_by_id("r_1714_REV").bounds = (0, glc_exch_rate)

wt_ec.model.reactions.get_by_id("r_1714").bounds = (-8.45, 0)
wt_ec.model.reactions.get_by_id("r_1714_REV").bounds = (0, 8.45)
wt_ec.model.reactions.get_by_id("r_1654").bounds = (-1.45, 0)
wt_ec.model.reactions.get_by_id("r_1654_REV").bounds = (0, 1.45)
wt_ec.ablation_result = wt_ec.ablate()
print(wt_ec.ablation_result)
ratio = wt_ec.get_ablation_ratio()
print(ratio)

breakpoint()

exch_rate_dict = {
    "r_1714": np.linspace(0, 2 * 8.45, 3),  # glucose
    "r_1654": np.linspace(0, 2 * 1.45, 3),  # ammonium
}
ratio_array, largest_component_array, growthrate_array = wt_ec.ablation_grid(
    exch_rate_dict
)
print(ratio_array)

breakpoint()

fig, ax = plt.subplots()
heatmap_ablation_grid(ax, exch_rate_dict, ratio_array, percent_saturation=True)
ax.set_xlabel("Glucose exchange (% max = 16.9)")
ax.set_ylabel("Ammonium exchange (% max = 2.9)")
plt.show()

breakpoint()
