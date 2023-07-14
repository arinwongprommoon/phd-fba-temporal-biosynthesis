#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns

from src.calc.matrix import get_susceptibility


@np.vectorize
def vget_growthrate(x):
    return x.ablated_flux[0]


filepath = "../data/interim/ec_grid_glc_amm.pkl"
with open(filepath, "rb") as handle:
    ablation_result_array = pickle.load(handle)


saturation_glc = 8.6869
saturation_amm = 1.4848

x_axis = np.linspace(0, 2 * saturation_glc, 32)
y_axis = np.linspace(0, 2 * saturation_amm, 32)

growthrate_array = vget_growthrate(ablation_result_array)
growthrate_array[0, :] = np.nan
growthrate_array[:, 0] = np.nan

breakpoint()

sus = get_susceptibility(growthrate_array, x_axis, y_axis)

sus_rot90 = get_susceptibility(np.rot90(growthrate_array), x_axis, y_axis[::-1])

diff0 = np.rot90(sus[0]) - sus_rot90[0]
diff1 = np.rot90(sus[1]) - sus_rot90[1]

breakpoint()
