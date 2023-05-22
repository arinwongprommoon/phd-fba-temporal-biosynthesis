#!/usr/bin/env python3

import matplotlib.pyplot as plt
from yeast8model import Yeast8Model, compare_fluxes, compare_ablation_times

y = Yeast8Model("./models/ecYeastGEM_batch.xml")
# y.knock_out_list(["YML120C"])
# y.knock_out_list(["YML120C", "foo"])

# y.add_media_components(["r_1761"])
# print(y.model.reactions.get_by_id("r_1761").bounds)

# y.make_auxotroph("BY4741")
# y.make_auxotroph("BY4742")
# y.make_auxotroph("BY4743")

# y.optimize()

y.ablation_result = y.ablate()
# fig, ax = plt.subplots()
# y.ablation_barplot(ax)
# plt.show()

z = Yeast8Model("./models/ecYeastGEM_batch.xml")
z.make_auxotroph("BY4741")

z.ablation_result = z.ablate()

fig, ax = plt.subplots()
compare_ablation_times(z.ablation_result, y.ablation_result, ax)
plt.show()

# dfs = compare_fluxes(y, z)

breakpoint()
