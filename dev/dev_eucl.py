#!/usr/bin/env python3

import numpy as np

from src.gem.yeast8model import Yeast8Model

glc_exch_rate = 16.89
wt_ec = Yeast8Model("../data/gemfiles/ecYeastGEM_batch_8-6-0.xml")
wt_ec.model.reactions.get_by_id("r_1714").bounds = (-glc_exch_rate, 0)
wt_ec.model.reactions.get_by_id("r_1714_REV").bounds = (0, glc_exch_rate)

# wt_ec.ablation_result = wt_ec.ablate()

exch_rate_dict = {
    "r_1714": np.linspace(0, 2 * 8.45, 3),  # glucose
    "r_1654": np.linspace(0, 2 * 1.45, 3),  # ammonium
}
ablation_result_array = wt_ec.euclidean_grid(exch_rate_dict)
print(ablation_result_array)

breakpoint()
