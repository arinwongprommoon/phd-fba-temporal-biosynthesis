#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns

from matplotlib import transforms


# Convenience functions
@np.vectorize
def vget_growthrate(x):
    return x.ablated_flux[0]


# Generate data
filepath = "../data/interim/ec_grid_glc_amm.pkl"
with open(filepath, "rb") as handle:
    ablation_result_array = pickle.load(handle)

saturation_glc = 8.6869
saturation_amm = 1.4848

x_axis = np.linspace(0, 2 * saturation_glc, 32)
y_axis = np.linspace(0, 2 * saturation_amm, 32)

growthrate_array = vget_growthrate(ablation_result_array)

# Remove weird parts of data
growthrate_array[0, :] = np.nan
growthrate_array[:, 0] = np.nan

# No modifications
fig, ax = plt.subplots()
sns.heatmap(data=growthrate_array, ax=ax)
ax.set_title("no mod")
plt.show()

# rot90
fig, ax = plt.subplots()
sns.heatmap(data=np.rot90(growthrate_array), ax=ax)
ax.set_title("rot90")
plt.show()

# flip about x
fig, ax = plt.subplots()
sns.heatmap(data=np.flip(growthrate_array, 1), ax=ax)
ax.set_title("flip about x")
plt.show()

breakpoint()
