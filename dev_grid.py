#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cobra

from yeast8model import Yeast8Model
from yeast8model import (
    heatmap_ablation_grid,
    get_ablation_ratio,
    get_ablation_largest_component,
    vget_ablation_ratio,
    vget_ablation_largest_component,
)

glc_exch_rate = 16.89
wt_ec = Yeast8Model("./models/ecYeastGEM_batch_8-6-0.xml")
wt_ec.model.reactions.get_by_id("r_1714").bounds = (-glc_exch_rate, 0)
wt_ec.model.reactions.get_by_id("r_1714_REV").bounds = (0, glc_exch_rate)

wt_ec.ablation_result = wt_ec.ablate()
print(wt_ec.ablation_result)
ratio = get_ablation_ratio(wt_ec.ablation_result)
largest_component = get_ablation_largest_component(wt_ec.ablation_result)
print(ratio)
print(largest_component)

breakpoint()

exch_rate_dict = {
    "r_1714": np.linspace(0, 2 * 8.45, 5),  # glucose
    "r_1654": np.linspace(0, 2 * 1.45, 5),  # ammonium
}
ablation_result_array = wt_ec.ablation_grid(exch_rate_dict)
print(ablation_result_array)

breakpoint()

ratio_array = vget_ablation_ratio(ablation_result_array)
largest_component_array = vget_ablation_largest_component(ablation_result_array)
ratio_gradient = np.gradient(ratio_array)

breakpoint()

fig, ax = plt.subplots()
heatmap_ablation_grid(ax, exch_rate_dict, ratio_array, percent_saturation=True)
ax.set_xlabel("Glucose exchange (% max = 16.9)")
ax.set_ylabel("Ammonium exchange (% max = 2.9)")
ax.set_title("Ratio")
plt.show()

breakpoint()

fig, ax = plt.subplots()
heatmap_ablation_grid(
    ax, exch_rate_dict, ratio_gradient[0], percent_saturation=True, vmin=None, vmax=None
)
ax.set_xlabel("Glucose exchange (% max = 16.9)")
ax.set_ylabel("Ammonium exchange (% max = 2.9)")
ax.set_title("Gradient, axis 0")
plt.show()

fig, ax = plt.subplots()
heatmap_ablation_grid(
    ax, exch_rate_dict, ratio_gradient[1], percent_saturation=True, vmin=None, vmax=None
)
ax.set_xlabel("Glucose exchange (% max = 16.9)")
ax.set_ylabel("Ammonium exchange (% max = 2.9)")
ax.set_title("Gradient, axis 1")
plt.show()
