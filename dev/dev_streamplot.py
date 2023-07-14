#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns

from matplotlib import transforms
from src.calc.matrix import get_susceptibility


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

# Compute susceptibility
# sus = get_susceptibility(growthrate_array, x_axis, y_axis)
sus = np.gradient(np.rot90(growthrate_array))
sus[0] = -sus[0]

# Draw susceptibility
fig, ax = plt.subplots()
sns.heatmap(data=sus[0], ax=ax)
ax.set_title("axis 0")
plt.show()

fig, ax = plt.subplots()
sns.heatmap(data=sus[1], ax=ax)
ax.set_title("axis 1")
plt.show()

# Prepare data for streamplot
# X, Y = np.meshgrid(x_axis, y_axis)
X, Y = np.meshgrid(np.linspace(0, 31, 32), np.linspace(0, 31, 32))
magnitudes = np.sqrt(sus[0] ** 2, sus[1] ** 2)

fig, ax = plt.subplots()
sns.heatmap(data=np.rot90(growthrate_array), ax=ax)
# ax.streamplot(X, Y, sus[0], sus[1], color=magnitudes, cmap="magma")
ax.quiver(X, Y, sus[1], sus[0])
ax.set_title("quiver")
plt.show()

breakpoint()

base = plt.gca().transData
rot = transforms.Affine2D().rotate_deg(270)
transl = transforms.Affine2D().translate(32, 0)
