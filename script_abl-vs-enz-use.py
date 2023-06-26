#!/usr/bin/env python3
#
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from yeast8model import Yeast8Model

plot_options = {
    "print_flux_extrema": False,
    "histogram": False,
    "subsystem_freqs": False,
    "subsystem_freqs/n": 100,
    "subsystem_sumfluxes": True,
}

# Load subsystems lookup
# Load subsystems.csv as a lookup table, convert to dict for speed
subsystem_df = pd.read_csv("easy_subsystems.csv", index_col=0)
subsystem_dict = dict(
    zip(subsystem_df.reaction_id.to_list(), subsystem_df.subsystem.to_list())
)


def extract_protein_ids(s):
    rxn_idx_list = s.index.to_list()
    enz_metabolite_ids = [
        rxn_idx.replace("draw_", "") + "[c]" for rxn_idx in rxn_idx_list
    ]
    return enz_metabolite_ids


def get_participating_rxn_ids(enz_metabolite_ids, s, ymodel):
    participating_rxn_ids = []
    enz_usage_fluxes = []
    for enz_metabolite_id in enz_metabolite_ids:
        enz_participating_rxns = list(
            ymodel.model.metabolites.get_by_id(enz_metabolite_id)._reaction
        )
        enz_participating_rxn_ids = [
            enz_participating_rxn.id for enz_participating_rxn in enz_participating_rxns
        ]
        participating_rxn_ids.extend(enz_participating_rxn_ids)
        enz_usage_fluxes.extend([s[idx]] * len(enz_participating_rxn_ids))
    return participating_rxn_ids, enz_usage_fluxes


def get_subsystem_list(participating_rxn_ids, subsystem_dict):
    subsystem_list = [
        subsystem_dict[rxn_id[:6]] if rxn_id[:2] == "r_" else "Enzyme usage"
        for rxn_id in participating_rxn_ids
    ]
    return subsystem_list


def make_subsystem_freq_table(subsystem_list):
    subsystem_freqs = pd.Series(subsystem_list).value_counts()
    return subsystem_freqs


def make_subsystem_sum_pivot(participating_rxn_ids, subsystem_list, enz_usage_fluxes):
    # Construct new DF
    participating_rxn_df = pd.DataFrame(
        {
            "participating_rxn_id": participating_rxn_ids,
            "subsystem": subsystem_list,
            "enz_usage_flux": enz_usage_fluxes,
        }
    )
    # Pivot table
    table = pd.pivot_table(
        participating_rxn_df, values="enz_usage_flux", index="subsystem", aggfunc=np.sum
    )
    table.drop(["Enzyme usage"], inplace=True)
    table = table.sort_values(ascending=False, by="enz_usage_flux")

    return table


def plot_subsystem_freqs(ymodel, s, ax):
    enz_metabolite_ids = extract_protein_ids(s)
    participating_rxn_ids, _ = get_participating_rxn_ids(enz_metabolite_ids, s, ymodel)
    # unique
    participating_rxn_ids = list(set(participating_rxn_ids))
    subsystem_list = get_subsystem_list(participating_rxn_ids, subsystem_dict)
    subsystem_freqs = make_subsystem_freq_table(subsystem_list)

    # plot
    subsystem_freqs.plot.barh(ax=ax)
    ax.invert_yaxis()
    ax.set_xlabel("Number of occurences of subsystem")


def plot_subsystem_sumfluxes(ymodel, s, ax):
    enz_metabolite_ids = extract_protein_ids(s)
    participating_rxn_ids, enz_usage_fluxes = get_participating_rxn_ids(
        enz_metabolite_ids, s, ymodel
    )
    subsystem_list = get_subsystem_list(participating_rxn_ids, subsystem_dict)
    table = make_subsystem_sum_pivot(
        participating_rxn_ids, subsystem_list, enz_usage_fluxes
    )

    # plot
    table.plot.barh(ax=ax)
    ax.invert_yaxis()
    ax.set_xlabel("Insert sensible x label here")


# Initialise model
glc_exch_rate = 16.89
wt_ec = Yeast8Model("./models/ecYeastGEM_batch_8-6-0.xml")
wt_ec.model.reactions.get_by_id("r_1714").bounds = (-glc_exch_rate, 0)
wt_ec.model.reactions.get_by_id("r_1714_REV").bounds = (0, glc_exch_rate)

# Ablate and store fluxes in each round
wt_ec.ablation_result = wt_ec.ablate()

ablation_fluxes = wt_ec.ablation_fluxes
ablation_fluxes_diff = ablation_fluxes.copy()
ablation_fluxes_diff.pop("original")
for biomass_component, fluxes in ablation_fluxes_diff.items():
    ablation_fluxes_diff[biomass_component] = (
        ablation_fluxes[biomass_component] - ablation_fluxes["original"]
    )
    if plot_options["print_flux_extrema"]:
        print(f"{biomass_component}")
        print(f"min {1e5 * ablation_fluxes_diff[biomass_component].min()} * 1e-5")
        print(f"max {1e5 * ablation_fluxes_diff[biomass_component].max()} * 1e-5")


if plot_options["histogram"]:
    # subplots
    # binrange=(-16e-5, ~90e-5) covers all diffs,
    #   but using a smaller range to emphasise the interesting part
    fig, ax = plt.subplots(nrows=len(ablation_fluxes_diff), ncols=1, sharex=True)
    for idx, (biomass_component, fluxes) in enumerate(ablation_fluxes_diff.items()):
        sns.histplot(
            fluxes * 1e5,
            bins=100,
            binrange=(-16, +20),
            log_scale=(False, True),
            ax=ax[idx],
        )
        ax[idx].set_title(biomass_component)
        ax[idx].set_xlabel("")
        ax[idx].set_ylabel("")
    # global labels
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel(r"Flux ($\times 10^{-5}$)")
    plt.ylabel("Number of reactions")
    plt.title("Changes in enzyme usage fluxes in biomass component ablation")
    plt.show()

if plot_options["subsystem_freqs"]:
    n = plot_options["subsystem_freqs/n"]
    for idx, (biomass_component, fluxes) in enumerate(ablation_fluxes_diff.items()):
        fig, ax = plt.subplots()
        s = fluxes.copy()
        s = s.sort_values(ascending=False)[:n]
        plot_subsystem_freqs(wt_ec, s, ax)
        ax.set_title(biomass_component)
    plt.show()

if plot_options["subsystem_sumfluxes"]:
    for idx, (biomass_component, fluxes) in enumerate(ablation_fluxes_diff.items()):
        fig, ax = plt.subplots(nrows=1, ncols=2)
        s = fluxes.copy()
        s_negative = s[s < 0]
        s_positive = s[s > 0]
        plot_subsystem_sumfluxes(wt_ec, s_positive, ax[0])
        ax[0].set_title(f"{biomass_component}, flux increases")
        plot_subsystem_sumfluxes(wt_ec, -s_negative, ax[1])
        ax[1].set_title(f"{biomass_component}, flux decreases")
    plt.show()

breakpoint()
