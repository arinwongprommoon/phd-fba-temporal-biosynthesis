#!/usr/bin/env python3

from yeast8model import Yeast8Model

y = Yeast8Model("./models/ecYeastGEM_batch.xml")
# y.knock_out_list(["YML120C"])
# y.knock_out_list(["YML120C", "foo"])
# y.add_media_components(["r_1761"])
# print(y.model.reactions.get_by_id("r_1761").bounds)
# y.make_auxotroph("BY4741")
# y.make_auxotroph("BY4742")
# y.make_auxotroph("BY4743")
# y.optimize()
df = y.ablate()
breakpoint()
