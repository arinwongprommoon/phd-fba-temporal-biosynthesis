#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns


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
growthrate_reciprocal = np.reciprocal(growthrate_array)
# growthrate_array = np.ones((32, 32))
growthrate_gradient = np.gradient(growthrate_array, x_axis, y_axis)

x_coeff_array = np.multiply(growthrate_reciprocal, x_axis[np.newaxis, :])
y_coeff_array = np.multiply(growthrate_reciprocal, y_axis[:, np.newaxis])

x_susceptibility = np.multiply(x_coeff_array, growthrate_gradient[0])
y_susceptibility = np.multiply(y_coeff_array, growthrate_gradient[1])

breakpoint()

growthrate_gradient_greater = np.abs(growthrate_gradient[0]) - np.abs(
    growthrate_gradient[1]
)
sns.heatmap(growthrate_gradient_greater, cmap="PuOr")
plt.show()

breakpoint()

susceptibility_greater = np.abs(x_susceptibility) - np.abs(y_susceptibility)
sns.heatmap(susceptibility_greater, cmap="PuOr")
plt.show()

breakpoint()
