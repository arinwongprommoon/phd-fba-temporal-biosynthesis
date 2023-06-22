#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

from yeast8model import Yeast8Model, piechart_ablation_grid

glc_exch_rate = 16.89
wt_ec = Yeast8Model("./models/ecYeastGEM_batch_8-6-0.xml")
wt_ec.model.reactions.get_by_id("r_1714").bounds = (-glc_exch_rate, 0)
wt_ec.model.reactions.get_by_id("r_1714_REV").bounds = (0, glc_exch_rate)

wt_ec.model.reactions.get_by_id("r_1714").bounds = (-8.45, 0)
wt_ec.model.reactions.get_by_id("r_1714_REV").bounds = (0, 8.45)
wt_ec.model.reactions.get_by_id("r_1654").bounds = (-1.45, 0)
wt_ec.model.reactions.get_by_id("r_1654_REV").bounds = (0, 1.45)

exch_rate_dict = {
    "r_1714": np.linspace(0, 2 * 8.45, 5),  # glucose
    "r_1654": np.linspace(0, 2 * 1.45, 5),  # ammonium
}
ablation_result_array = wt_ec.ablation_grid(exch_rate_dict)

breakpoint()

piechart_ablation_grid(exch_rate_dict, ablation_result_array, True)
plt.show()

breakpoint()
