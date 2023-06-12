#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cobra

from yeast8model import Yeast8Model, biomass_component_list_orig
import time

penalty_coeff_range = np.linspace(0, 1, 3)

# Define data structure to store investigation results.

# Alternative considered: dict with floats as keys.
# Shot that down because I think that's weird and not the point of a dict,
# especially if the penalty_coeff_range can change.
effect_list = [
    {"penalty_coeff": penalty_coeff, "ablation_result": None}
    for penalty_coeff in penalty_coeff_range
]

for effect_item in effect_list:
    print(effect_item)

    print(f"coeff {effect_item['penalty_coeff']}")
    start = time.time()
    y = Yeast8Model("./models/yeast-GEM_8-6-0.xml")
    glucose_bounds = (-4.75, 0)  # gives a sensible growth rate for wt
    y.add_media_components(["r_1992"])
    y.model.reactions.r_1714.bounds = glucose_bounds
    y.biomass_component_list = biomass_component_list_orig

    sol_orig = y.optimize()

    penalty_coefficient = effect_item["penalty_coeff"]
    y.set_flux_penalty(penalty_coefficient=penalty_coefficient)
    end = time.time()
    print(f"penalty set with coeff.  elapsed time: {end - start} s")

    sol_pen = y.optimize()  # check if this line is necessary

    ablation_result = y.ablate(verbose=False)
    effect_item["ablation_result"] = ablation_result

print(effect_list)
