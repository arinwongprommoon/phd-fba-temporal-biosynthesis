#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cobra

from yeast8model import Yeast8Model, piechart_ablation_grid
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

wt_ec.model.reactions.get_by_id("r_1714").bounds = (-8.45, 0)
wt_ec.model.reactions.get_by_id("r_1714_REV").bounds = (0, 8.45)
wt_ec.model.reactions.get_by_id("r_1654").bounds = (-1.45, 0)
wt_ec.model.reactions.get_by_id("r_1654_REV").bounds = (0, 1.45)

exch_rate_dict = {
    "r_1714": np.linspace(0, 2 * 8.45, 3),  # glucose
    "r_1654": np.linspace(0, 2 * 1.45, 3),  # ammonium
}
ablation_result_array = wt_ec.ablation_grid(exch_rate_dict)

breakpoint()

fig, ax = plt.subplots()
piechart_ablation_grid(ax, exch_rate_dict, ablation_result_array)
plt.show()

breakpoint()
