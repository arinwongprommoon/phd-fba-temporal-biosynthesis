#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from yeast8model import (
    Yeast8Model,
    compare_fluxes,
    compare_ablation_times,
    get_exch_saturation,
    heatmap_ablation_grid,
)

y = Yeast8Model("./models/ecYeastGEM_batch_8-6-0.xml")
# z = Yeast8Model("./models/ecYeastGEMfull.yml")
# y.knock_out_list(["YML120C"])
# y.knock_out_list(["YML120C", "foo"])

# y.add_media_components(["r_1761"])
# print(y.model.reactions.get_by_id("r_1761").bounds)

# y.make_auxotroph("BY4741")
# y.make_auxotroph("BY4742")
# y.make_auxotroph("BY4743")

# y.optimize()

# r = y.get_ablation_ratio()
# print(r)
exch_rate_dict = {
    "r_1714": np.linspace(0, 18, 2),
    "r_1654": np.linspace(0, 18, 2),
}
ra, la, gra = y.ablation_grid(exch_rate_dict)
# breakpoint()

fig, ax = plt.subplots()
heatmap_ablation_grid(ax, exch_rate_dict, ra, la, percent_saturation=True)
plt.show()

fig, ax = plt.subplots()
heatmap_ablation_grid(ax, exch_rate_dict, gra, la, percent_saturation=True)
plt.show()
# # y.ablation_barplot(ax)

# z = Yeast8Model("./models/ecYeastGEM_batch.xml")
# z.make_auxotroph("BY4741")

# z.ablation_result = z.ablate()

# fig, ax = plt.subplots()
# compare_ablation_times(z.ablation_result, y.ablation_result, ax)
# plt.show()

# dfs = compare_fluxes(y, z)

# grs = get_exch_saturation(y, "r_1714", np.linspace(0, 18.6, 10))

breakpoint()
