#!/usr/bin/env python3
#
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.gem.yeast8model import Yeast8Model

plot_options = {
    # Fraction of original sum of absolute values of fluxes to constrain fluxes
    # to.  If None, do not impose this constraint.  Otherwise, supply a float
    # from 0 to 1.
    "constrain_fluxes": None,
    # When computing how enzyme usage fluxes change during each round of
    # ablation, print mininum (greatest magnitude of negative flux change) and
    # maximum (greatest magnitude of positive flux change) when each biomass
    # component is prioritised.
    "print_flux_extrema": False,
    # Draw a histogram showing how these fluxes changes.  Horizontal axis = flux
    # change (binned), vertical axis = how many reactions (log scale).
    "histogram": False,
    # Take the top N greatest-magnitude (negative and positive) enzyme usage
    # flux changes, see which reactions these enzymes correspond to, and count
    # how many times each subsystem is represented.  Draw bar plots for each
    # biomass component.
    "subsystem_freqs": False,
    # N for the above.
    "subsystem_freqs/n": 100,
    # For each of negative and positive enzyme usage flux changes, see which
    # reactions these enzymes correspond to, and sum all the flux changes for
    # each subsystem.  Draw bar plots for each biomass component.
    "subsystem_sumfluxes": False,
    # Draw a heatmap, columns showing the biomass component, rows showing
    # each enzyme, and colours showing flux changes.  Rows are grouped by sub-
    # system, alphabetically, and are labelled by subsystem.
    "subsystem_heatmap": True,
    # Tolerance value for heatmap -- heatmap excludes flux difference magnitudes
    # that are below this value.  Useful for excluding lots of 'uninteresting'
    # fluxes.
    "subsystem_heatmap/tol": 3e-6,
}

# Load subsystems lookup
# Load subsystems.csv as a lookup table, convert to dict for speed
subsystem_df = pd.read_csv("../data/lookup/easy_subsystems.csv", index_col=0)
subsystem_dict = dict(
    zip(subsystem_df.reaction_id.to_list(), subsystem_df.subsystem.to_list())
)


def extract_protein_ids(s):
    """Extract protein metabolite IDs (GECKO) from a list of draw_prot reactions."""
    rxn_idx_list = s.index.to_list()
    enz_metabolite_ids = [
        rxn_idx.replace("draw_", "") + "[c]" for rxn_idx in rxn_idx_list
    ]
    return enz_metabolite_ids


def get_participating_rxn_ids(enz_metabolite_ids, s, ymodel):
    """Get IDs of reactions that a list of enzymes participates in.

    Parameters
    ----------
    enz_metabolite_ids : list of str
        List of enzyme metabolite IDs (GECKO), in the form of "protein_XXXX[c]".
    s : pandas.Series
        List of fluxes, each associated with a "draw_prot_XXXX" reaction.
    ymodel : yeast8model.Yeast8Model object
        Model object, needed for the reaction list

    Returns
    -------
    participating_rxn_ids : list of str
        List of participating reaction IDs.  This list may have more elements
        than enz_metabolite_ids.
    enz_usage_fluxes : list of float
        List of fluxes.  This list has the same number of elements as
        participating_rxn_ids.
    """
    participating_rxn_ids = []
    enz_usage_fluxes = []
    for idx, enz_metabolite_id in enumerate(enz_metabolite_ids):
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
    """Get list of subsystems based on a list of reaction IDs."""
    subsystem_list = [
        subsystem_dict[rxn_id[:6]] if rxn_id[:2] == "r_" else "Enzyme usage"
        for rxn_id in participating_rxn_ids
    ]
    return subsystem_list


def make_subsystem_freq_table(subsystem_list):
    """Make a frequency table of subsystems based on a list of subsystems."""
    subsystem_freqs = pd.Series(subsystem_list).value_counts()
    return subsystem_freqs


def make_subsystem_sum_pivot(participating_rxn_ids, subsystem_list, enz_usage_fluxes):
    """Make a pivot table that lists the sum of fluxes for each subsystem."""
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
    """Draw bar plot showing occurences of each subsystem in a list of reactions."""
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
    """Draw bar plot showing sum of fluxes corresponding to each subsystem"""
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
    ax.set_xlabel("Sum of fluxes")


if __name__ == "__main__":
    # Initialise model
    glc_exch_rate = 16.89
    wt_ec = Yeast8Model("../data/gemfiles/ecYeastGEM_batch_8-6-0.xml")
    wt_ec.model.reactions.get_by_id("r_1714").bounds = (-glc_exch_rate, 0)
    wt_ec.model.reactions.get_by_id("r_1714_REV").bounds = (0, glc_exch_rate)

    # Constrain fluxes
    if plot_options["constrain_fluxes"] is not None:
        sol = wt_ec.optimize()
        orig_flux_sum = sol.fluxes.abs().sum()
        ub = plot_options["constrain_fluxes"] * orig_flux_sum
        wt_ec.set_flux_constraint(upper_bound=ub)
        sol = wt_ec.optimize()

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
        plt.tick_params(
            labelcolor="none", top=False, bottom=False, left=False, right=False
        )
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
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
            s = fluxes.copy()
            s_negative = s[s < 0]
            s_positive = s[s > 0]
            plot_subsystem_sumfluxes(wt_ec, s_positive, ax[0])
            ax[0].set_title(f"{biomass_component}, flux increases")
            plot_subsystem_sumfluxes(wt_ec, -s_negative, ax[1])
            ax[1].set_title(f"{biomass_component}, flux decreases")
            fig.tight_layout()
        plt.show()

    if plot_options["subsystem_heatmap"]:
        list_participating_rxn_df = []
        # Make DF for each biomass component
        for idx, (biomass_component, fluxes) in enumerate(ablation_fluxes_diff.items()):
            # get fluxes
            s = fluxes.copy()
            # get data needed for DF
            enz_metabolite_ids = extract_protein_ids(s)
            participating_rxn_ids, enz_usage_fluxes = get_participating_rxn_ids(
                enz_metabolite_ids, s, wt_ec
            )
            subsystem_list = get_subsystem_list(participating_rxn_ids, subsystem_dict)
            # construct DF
            enz_usage_flux_column = "enz_usage_flux_" + biomass_component
            participating_rxn_df = pd.DataFrame(
                {
                    "participating_rxn_id": participating_rxn_ids,
                    "subsystem": subsystem_list,
                    enz_usage_flux_column: enz_usage_fluxes,
                }
            )
            list_participating_rxn_df.append(participating_rxn_df)
        # construct master DF with info from all biomass components
        left_columns = list_participating_rxn_df[0].iloc[:, 0:2]
        enz_usage_columns = pd.concat(
            [
                list_participating_rxn_df[idx].iloc[:, -1]
                for idx in range(len(list_participating_rxn_df))
            ],
            axis=1,
        )
        all_fluxes_df = pd.concat([left_columns, enz_usage_columns], axis=1)
        # Drop enzyme usage reactions
        all_fluxes_df = all_fluxes_df[all_fluxes_df.subsystem != "Enzyme usage"]
        # Deal with duplicate reaction IDs by summing the fluxes
        all_fluxes_df = all_fluxes_df.groupby(
            ["participating_rxn_id", "subsystem"], as_index=False
        ).sum(numeric_only=True)
        # Sort alphabetically by subsystem
        all_fluxes_df = all_fluxes_df.sort_values(by=["subsystem"])

        # Drop fluxes with magnitude below a certain tol value
        enz_usage_colnames = [
            colname
            for colname in all_fluxes_df.columns.to_list()
            if colname.startswith("enz_usage_flux_")
        ]
        tol = plot_options["subsystem_heatmap/tol"]
        farzero_fluxes_df = all_fluxes_df.loc[
            (all_fluxes_df[enz_usage_colnames].abs() >= tol).any(axis=1)
        ]
        print(f"heatmap has {len(farzero_fluxes_df)} rows")

        # Define y-labels for heatmap: subsystem names
        subsystem_list_sorted = farzero_fluxes_df.subsystem.to_list()
        # Replace duplicates with space so that the subsystem name is only
        # present the first time it occurs
        subsystem_labels = []
        compare_string = ""
        for subsystem_string in subsystem_list_sorted:
            if subsystem_string == compare_string:
                subsystem_string = " "
            else:
                compare_string = subsystem_string
            subsystem_labels.append(subsystem_string)

        # Draw 2d heatmap
        fig, ax = plt.subplots(figsize=(10, 25))
        sns.heatmap(
            farzero_fluxes_df.iloc[:, 2:] * 1e4,
            xticklabels=list(ablation_fluxes_diff.keys()),
            yticklabels=subsystem_labels,
            center=0,
            robust=True,
            cmap="PiYG",
            cbar_kws={
                "label": r"Flux change [$\times 10^{-4} \mathrm{protein} mmol \cdot g_{DW}^{-1}$]"
            },
            ax=ax,
        )
        ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        ax.set_title(
            f"Enzyme usage changes as a result of ablation\n(magnitudes smaller than {tol} excluded)"
        )
        plt.show()

pdf_filename = "abl_vs_enz_use_plots.pdf"
with PdfPages(pdf_filename) as pdf:
    for fig in range(1, plt.gcf().number + 1):
        pdf.savefig(fig)
