#!/usr/bin/env python3

from src.calc.ablation import get_ablation_ratio, get_custom_ablation_ratio
from src.gem.yeast8model import Yeast8Model

glc_exch_rate = 16.89
wt_ec = Yeast8Model("../data/gemfiles/ecYeastGEM_batch_8-6-0.xml")
wt_ec.model.reactions.get_by_id("r_1714").bounds = (-glc_exch_rate, 0)
wt_ec.model.reactions.get_by_id("r_1714_REV").bounds = (0, glc_exch_rate)
ablation_result = wt_ec.ablate()

breakpoint()

r1 = get_custom_ablation_ratio(ablation_result, ["protein"])
r2 = get_custom_ablation_ratio(ablation_result, ["protein", "carbohydrate"])
r3 = get_ablation_ratio(ablation_result)

print(r1)
print(r2)
print(r3)

breakpoint()
