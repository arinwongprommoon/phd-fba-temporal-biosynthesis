#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns

from matplotlib import transforms
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

sus = get_susceptibility(growthrate_array, x_axis, y_axis)

# X, Y = np.meshgrid(x_axis, y_axis)
X, Y = np.meshgrid(np.linspace(0, 31, 32), np.linspace(0, 31, 32))
magnitudes = np.sqrt(sus[0] ** 2, sus[1] ** 2)

base = plt.gca().transData
rot = transforms.Affine2D().rotate_deg(270)
transl = transforms.Affine2D().translate(32, 0)

fig, ax = plt.subplots()
data = np.random.randn(100)
ax.plot(data, transform=rot + base)
# ax.streamplot(
#     X, Y, sus[0], sus[1], transform=rot + base, color=magnitudes, cmap="magma"
# )
plt.show()

breakpoint()
