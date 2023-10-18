#!/usr/bin/env python3
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from src.gem.yeast8model import Yeast8Model

# Construct models of strains
glc_exch_rate = 16.89

ymodels = {
    "wt": None,
    "BY4741": None,
    "zwf1": None,
    "tsa2": None,
}

ymodels["wt"] = Yeast8Model("../data/gemfiles/ecYeastGEM_batch_8-6-0.xml")
ymodels["wt"].model.reactions.get_by_id("r_1714").bounds = (-glc_exch_rate, 0)
ymodels["wt"].model.reactions.get_by_id("r_1714_REV").bounds = (0, glc_exch_rate)
ymodels["wt"].solution = ymodels["wt"].optimize()

ymodels["BY4741"] = Yeast8Model("../data/gemfiles/ecYeastGEM_batch_8-6-0.xml")
ymodels["BY4741"].make_auxotroph("BY4741")
ymodels["BY4741"].model.reactions.get_by_id("r_1714").bounds = (-glc_exch_rate, 0)
ymodels["BY4741"].model.reactions.get_by_id("r_1714_REV").bounds = (0, glc_exch_rate)
ymodels["BY4741"].solution = ymodels["BY4741"].optimize()

ymodels["zwf1"] = Yeast8Model("../data/gemfiles/ecYeastGEM_batch_8-6-0.xml")
ymodels["zwf1"].make_auxotroph("BY4741")
ymodels["zwf1"].knock_out_list(["YNL241C"])
ymodels["zwf1"].model.reactions.get_by_id("r_1714").bounds = (-glc_exch_rate, 0)
ymodels["zwf1"].model.reactions.get_by_id("r_1714_REV").bounds = (0, glc_exch_rate)
ymodels["zwf1"].solution = ymodels["zwf1"].optimize()

ymodels["tsa2"] = Yeast8Model("../data/gemfiles/ecYeastGEM_batch_8-6-0.xml")
ymodels["tsa2"].make_auxotroph("BY4742")
ymodels["tsa2"].knock_out_list(["YDR453C"])
ymodels["tsa2"].model.reactions.get_by_id("r_1714").bounds = (-glc_exch_rate, 0)
ymodels["tsa2"].model.reactions.get_by_id("r_1714_REV").bounds = (0, glc_exch_rate)
ymodels["tsa2"].solution = ymodels["tsa2"].optimize()

# Ablate and save
ablation_results = {
    "wt": None,
    "BY4741": None,
    "zwf1": None,
    "tsa2": None,
}

for model_key, _ in ablation_results.items():
    ymodels[model_key].ablation_result = ymodels[model_key].ablate()
    ablation_results[model_key] = ymodels[model_key].ablation_result

# Work
component_list = ablation_results["wt"].priority_component.to_list()[1:]
ymodel_keys = ablation_results.keys()

# Construct data structures for bar plots
# Hard-coding positions because honestly, it is easier.
est_times_dict = {}
Tpar_dict = {}
for key in ymodel_keys:
    est_times_dict[key] = ablation_results[key].iloc[1:, 2].to_list()
    Tpar_dict[key] = [ablation_results[key].iloc[2, 3]]

est_times_df = pd.DataFrame(est_times_dict)
est_times_df.index = component_list
Tpar_df = pd.DataFrame(Tpar_dict)
Tpar_df.index = ["Tpar"]

# Draw

prop_cycle = plt.rcParams["axes.prop_cycle"]
default_mpl_colors = prop_cycle.by_key()["color"]

# https://stackoverflow.com/a/69130629
fig, ax = plt.subplots()

est_times_array = est_times_df.T.to_numpy()
Tpar_array = Tpar_df.T.to_numpy()
x_pos = np.arange(len(ymodel_keys))

for i in range(est_times_array.shape[1]):
    bottom = np.sum(est_times_array[:, 0:i], axis=1)
    est_times_bars = ax.bar(
        x_pos - 0.2,
        est_times_array[:, i],
        bottom=bottom,
        width=0.3,
    )

for i in range(Tpar_array.shape[1]):
    bottom = np.sum(Tpar_array[:, 0:i], axis=1)
    est_times_bars = ax.bar(x_pos + 0.2, Tpar_array[:, i], bottom=bottom, width=0.3)

ax.set_xticks(x_pos)
ax.set_xticklabels(["Wild type", "BY4741", "zwf1Δ", "tsa2Δ"])
ax.set_xlabel("Strain")
ax.set_ylabel("Estimated time (h)")

# Legend: colour = biomass component
handles = []
# assumes that default_mpl_colors (usually 10 elements) is longer than
# component_list (usually 7 elements)
# Using colour patches ref:
# https://matplotlib.org/stable/tutorials/intermediate/legend_guide.html#creating-artists-specifically-for-adding-to-the-legend-aka-proxy-artists
for color, component in zip(default_mpl_colors[: len(component_list)], component_list):
    color_patch = mpatches.Patch(color=color, label=component)
    handles.append(color_patch)
grey_patch = mpatches.Patch(color=default_mpl_colors[7], label="Tpar")
handles.append(grey_patch)
fig.legend(handles, component_list + [r"$T_{par}$"], loc="center right")
fig.subplots_adjust(right=0.75)

filename = "combined_barplot"
# Save all open figures to PDF
pdf_filename = "../reports/" + filename + ".pdf"
with PdfPages(pdf_filename) as pdf:
    for fig in range(1, plt.gcf().number + 1):
        pdf.savefig(fig)
# Close all figures
plt.close("all")
