#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import pickle

from matplotlib.backends.backend_pdf import PdfPages
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale


model_options = {
    # "glc" or "pyr"
    "carbon_source": "pyr",
}

# Load saved data
usg_filename = "ec_usg_" + model_options["carbon_source"] + "_amm"
usg_filepath = "../data/interim/" + usg_filename + ".pkl"
with open(usg_filepath, "rb") as handle:
    ablation_result_array = pickle.load(handle)

# Adjust data variable dimensions
ablation_result_1d = ablation_result_array.ravel()
multicond_enz_use_array = np.concatenate(ablation_result_1d)
multicond_enz_use_array.shape

# Scale
scaled_array = scale(multicond_enz_use_array)

# PCA
pca = PCA()
Xt = pca.fit_transform(scaled_array)
pca1 = Xt[:, 0]
pca2 = Xt[:, 1]

# Check explained variance
print(np.cumsum(pca.explained_variance_ratio_))

# Color dots by biomass components, using the default cycle.
# Original = C0, lipid = C1, etc.
num_components = 8
color_dict = dict(
    zip(list(range(num_components)), ["C" + str(num) for num in range(num_components)])
)
color_list = [color_dict[el] for el in (np.arange(len(pca1)) % num_components)]

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(pca1, pca2, color=color_list)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")

# Plot each condition
num_conds = int(len(pca1) / num_components)
axis_size = int(np.sqrt(num_conds))
color_list = [color_dict[el] for el in range(num_components)]

if model_options["carbon_source"] == "glc":
    exch_rate_dict = {
        "r_1714": np.linspace(0.5 * 8.6869, 2 * 8.6869, 4),  # glucose
        "r_1654": np.linspace(0.5 * 1.4848, 2 * 1.4848, 4),  # ammonium
    }
    x_axis = exch_rate_dict["r_1714"]
elif model_options["carbon_source"] == "pyr":
    exch_rate_dict = {
        "r_2033": np.linspace(0.5 * 4.4444, 2 * 4.4444, 4),  # pyruvate
        "r_1654": np.linspace(0.5 * 1.0, 2 * 1.0, 4),  # ammonium
    }
    x_axis = exch_rate_dict["r_2033"]
else:
    print("Error: No carbon source")
y_axis = exch_rate_dict["r_1654"]

fig, ax = plt.subplots(ncols=axis_size, nrows=axis_size, figsize=(12, 12))
for cond in range(num_conds):
    x_pos = cond // axis_size
    y_pos = cond % axis_size
    start_idx = cond * num_components
    ax[x_pos, y_pos].scatter(
        pca1[start_idx : start_idx + num_components],
        pca2[start_idx : start_idx + num_components],
        color=color_list,
        marker="+",
        s=40,
    )
    ax[x_pos, y_pos].set_xlim(np.min(pca1), np.max(pca1))
    ax[x_pos, y_pos].set_ylim(np.min(pca2), np.max(pca2))
    ax[x_pos, y_pos].tick_params(
        axis="both", bottom=False, left=False, labelbottom=False, labelleft=False
    )

    c_exch = x_axis[x_pos]
    n_exch = y_axis[y_pos]
    ax[x_pos, y_pos].set_title(f"C {c_exch:.2f}, N {n_exch:.2f}")

# Save all open figures to PDF
filename = "pca_" + model_options["carbon_source"]
pdf_filename = "../reports/" + filename + ".pdf"
with PdfPages(pdf_filename) as pdf:
    for fig in range(1, plt.gcf().number + 1):
        pdf.savefig(fig)
# Close all figures
plt.close("all")
