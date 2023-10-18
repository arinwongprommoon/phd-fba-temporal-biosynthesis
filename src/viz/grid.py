#!/usr/bin/env python3

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

prop_cycle = plt.rcParams["axes.prop_cycle"]
default_mpl_colors = prop_cycle.by_key()["color"]


def heatmap_ablation_grid(
    ax,
    exch_rate_dict,
    ratio_array,
    largest_component_array=None,
    percent_saturation=False,
    saturation_point=(None, None),
    saturation_grid=False,
    vmin=0,
    vmax=2,
    showticklabels=True,
    center=None,
    cmap="RdBu_r",
    cbar_label="ratio",
):
    """Draw heatmap from 2d ablation grid

    Parameters
    ----------
    ax : matplotlib.pyplot.Axes object
        Axes to draw heatmap on.
    exch_rate_dict : dict
        dict that stores the two exchange reactions to vary and the uptake
        rate values to use.  It should be in this format:

        d = {
            'r_exch_rxn_1' : <array-like>,
            'r_exch_rxn_2' : <array-like>,
            }

    ratio_array : numpy.ndarray (2-dimensional)
        Array of ablation ratios, output from ablation_grid()
    largest_component_array : numpy.ndarray (2-dimensional), optional
        Array of largest biomass components, output from ablation_grid()
    percent_saturation : bool, optional
        Whether to scale axis labels so that the numbers displayed are percent
        of saturation.  Default False.
    saturation_point : (float, float) tuple, optional
        Values of exchange fluxes to use as saturation.  If either element is
        None, use the max as saturation.
    saturation_grid : bool, optional
        Whether to draw grid lines to show where saturation is.  Default False.
    vmin : float, optional
        Minimum of range for colour bar.  Default 0.
    vmax : float, optional
        Maximum of range for colour bar.  Default 2.
    showticklabels : bool, optional
        Whether to show x/y-axis tick labels.  Default True.
    cbar_label : string, optional
        Label for colour bar.  Default "ratio".

    Examples
    --------
    FIXME: Add docs.

    """
    # If largest_component_array is supplied, use it as text labels on heatmap.
    # This design takes advantage of seaborn.heatmap(annot=None) being default.
    if largest_component_array is None:
        annot_input = largest_component_array
    # TODO: Improve error-handling by checking if this is a 2D numpy array
    else:
        annot_input = np.rot90(largest_component_array)

    # Define x & y tick labels
    heatmap_xticklabels = list(exch_rate_dict.values())[0].copy()
    heatmap_yticklabels = list(exch_rate_dict.values())[1][::-1].copy()
    saturation_x = saturation_point[0]
    saturation_y = saturation_point[1]
    # ... depending on saturation-related arguments
    if percent_saturation:
        if saturation_x is not None:
            heatmap_xticklabels /= saturation_x
        else:
            heatmap_xticklabels /= np.max(heatmap_xticklabels)
        heatmap_xticklabels *= 100

        if saturation_y is not None:
            heatmap_yticklabels /= saturation_y
        else:
            heatmap_yticklabels /= np.max(heatmap_yticklabels)
        heatmap_yticklabels *= 100

        # and draw grid lines if specified
        # This only makes sense if percent_saturation is True.
        if saturation_grid:
            ax.axvline(
                np.searchsorted(heatmap_xticklabels, 100, side="left"), color="k"
            )
            # doing this because y axis is defined 'in reverse' & to have line
            # position consistent with x axis
            ax.axhline(
                np.searchsorted(heatmap_yticklabels[::-1], 100, side="right"), color="k"
            )

    # Draws heatmap.
    # Rounding directly on the x/yticklabels variables because of known
    # matplotlib-seaborn bug:
    # - https://github.com/mwaskom/seaborn/issues/1005
    # - https://stackoverflow.com/questions/63964006/round-decimal-places-seaborn-heatmap-labels
    # - https://stackoverflow.com/questions/50571592/matplotlib-formatstrformatter-returns-wrong-values
    sns.heatmap(
        data=np.rot90(ratio_array),
        annot=annot_input,
        xticklabels=np.around(heatmap_xticklabels, decimals=1),
        yticklabels=np.around(heatmap_yticklabels, decimals=1),
        robust=True,
        vmin=vmin,
        vmax=vmax,
        center=center,
        cmap=cmap,
        cbar_kws={"label": cbar_label},
        fmt="",
        ax=ax,
    )
    ax.set_xlabel(list(exch_rate_dict.keys())[0])
    ax.set_ylabel(list(exch_rate_dict.keys())[1])

    # Hide tick labels
    if not showticklabels:
        ax.set(xticklabels=[])
        ax.set(yticklabels=[])


def piechart_ablation_grid(
    exch_rate_dict,
    ablation_result_array,
    percent_saturation=False,
    xlabel=None,
    ylabel=None,
):
    """Grid of pie charts showing proportions of prioritised components in ablated-predicted time

    Draws a grid of pie charts.  Each pie chart shows the proportions of the
    times predicted for prioritising each biomass component by ablation study.
    x and y axes show the corresponding exchange reaction fluxes.

    If a pie chart at any position cannot be drawn -- e.g. when exchange rate is
    0 -- then that position shows the text 'N/A'.

    Parameters
    ----------
    exch_rate_dict : dict
        dict that stores the two exchange reactions to vary and the uptake
        rate values to use.  It should be in this format:

        d = {
            'r_exch_rxn_1' : <array-like>,
            'r_exch_rxn_2' : <array-like>,
            }

    ablation_result_array : 2-dimensional numpy.ndarray of objects
        Array of ablation result DataFrames.
        Indexing follows the exchange reaction
        arrays in the input exch_rate_dict, i.e. ratio_array[x][y]
        corresponds to exch_rate_dict['r_exch_rxn_1'][x] and
        exch_rate_dict['r_exch_rxn_2'][y].
    percent_saturation : bool, optional
        Whether to scale axis labels so that the numbers displayed are percent
        of the highest value of the axis (usually saturation).  Default False.
    xlabel : str
        x-axis label.  Defaults to name of first exchange reaction.
    ylabel : str
        y-axis label.  Defaults to name of second exchange reaction.

    Examples
    --------
    # Initialise model
    wt_ec = Yeast8Model(...)

    # Create ablation grid
    exch_rate_dict = {
        "r_1714": np.linspace(0, 2 * 8.45, 3),  # glucose
        "r_1654": np.linspace(0, 2 * 1.45, 3),  # ammonium
    }
    ablation_result_array = wt_ec.ablation_grid(exch_rate_dict)

    # Draw pie chart
    piechart_ablation_grid(exch_rate_dict, ablation_result_array)
    plt.show()
    """
    ablation_result_array = np.rot90(ablation_result_array)

    nrows, ncols = ablation_result_array.shape
    # Have exch_rate values in an extra column & row
    nrows += 1
    ncols += 1
    fig, ax = plt.subplots(nrows, ncols)

    # Define labels for legend
    # Debt: Assumes that all ablation_result DataFrames share the same format
    ablation_result_temp = ablation_result_array[0, 0]
    component_list = ablation_result_temp.priority_component.to_numpy().T
    # deletes 'original' priority component
    component_list = np.delete(component_list, 0)
    component_list = component_list.tolist()

    # Define axis labels
    global_xaxislabels = list(exch_rate_dict.values())[0]
    # Scale if specified
    global_yaxislabels = list(exch_rate_dict.values())[1][::-1]
    if percent_saturation:
        global_xaxislabels /= np.max(global_xaxislabels)
        global_xaxislabels *= 100
        global_yaxislabels /= np.max(global_yaxislabels)
        global_yaxislabels *= 100
    # Dummy value for corner position -- not used
    # (could be useful for debugging)
    global_xaxislabels = np.append([-1], global_xaxislabels)
    global_yaxislabels = np.append(global_yaxislabels, [-1])

    # Draw pie charts
    for row_idx, global_yaxislabel in enumerate(global_yaxislabels):
        # Left column reserved for exch rate 2 labels
        ax[row_idx, 0].set_axis_off()
        # Bottom left corner must be blank
        if row_idx == len(global_yaxislabels) - 1:
            pass
        else:
            # Print exch rate label
            ax[row_idx, 0].text(
                x=0.5,
                y=0.5,
                s=f"{global_yaxislabel:.3f}",
                ha="center",
                va="center",
            )
        for col_idx, global_xaxislabel in enumerate(global_xaxislabels):
            # Bottom row reserved for exch rate 1 labels
            if row_idx == len(global_yaxislabels) - 1:
                ax[row_idx, col_idx].set_axis_off()
                # Bottom left corner must be blank
                if col_idx == 0:
                    pass
                else:
                    # Print exch rate label
                    ax[row_idx, col_idx].text(
                        x=0.5,
                        y=0.5,
                        s=f"{global_xaxislabel:.3f}",
                        ha="center",
                        va="center",
                    )
            else:
                # Left column reserved for exch rate 2 labels
                if col_idx == 0:
                    pass
                else:
                    # Get times
                    ablation_result = ablation_result_array[row_idx, col_idx - 1]
                    ablation_times_df = ablation_result.loc[
                        ablation_result.priority_component != "original",
                        ablation_result.columns == "ablated_est_time",
                    ]
                    ablation_times = ablation_times_df.to_numpy().T[0]
                    # Deal with edge cases, e.g. negative values when exch rate is 0
                    try:
                        artists = ax[row_idx, col_idx].pie(ablation_times)
                    except:
                        print(f"Unable to draw pie chart at [{row_idx}, {col_idx}].")
                        ax[row_idx, col_idx].set_axis_off()
                        ax[row_idx, col_idx].text(
                            x=0.5,
                            y=0.5,
                            s="N/A",
                            ha="center",
                            va="center",
                            fontweight="bold",
                        )

    # For global axis labels: create a big subplot and hide everything except
    # for the labels
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    if xlabel is None:
        xlabel = list(exch_rate_dict.keys())[0]
    if ylabel is None:
        ylabel = list(exch_rate_dict.keys())[1]
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # Legend: colour = biomass component
    fig.legend(artists[0], component_list, loc="center right")
    fig.subplots_adjust(right=0.75)


def barchart_ablation_grid(
    exch_rate_dict,
    ablation_result_array,
    percent_saturation=False,
    xlabel=None,
    ylabel=None,
):
    """Grid of bar charts showing times of prioritised components in ablated-predicted time

    Draws a grid of bar charts.  Each bar chart shows the
    times predicted for prioritising each biomass component by ablation study.
    x and y axes show the corresponding exchange reaction fluxes.

    If a bar chart at any position cannot be drawn -- e.g. when exchange rate is
    0 -- then that position shows the text 'N/A'.

    Parameters
    ----------
    exch_rate_dict : dict
        dict that stores the two exchange reactions to vary and the uptake
        rate values to use.  It should be in this format:

        d = {
            'r_exch_rxn_1' : <array-like>,
            'r_exch_rxn_2' : <array-like>,
            }

    ablation_result_array : 2-dimensional numpy.ndarray of objects
        Array of ablation result DataFrames.
        Indexing follows the exchange reaction
        arrays in the input exch_rate_dict, i.e. ratio_array[x][y]
        corresponds to exch_rate_dict['r_exch_rxn_1'][x] and
        exch_rate_dict['r_exch_rxn_2'][y].
    percent_saturation : bool, optional
        Whether to scale axis labels so that the numbers displayed are percent
        of the highest value of the axis (usually saturation).  Default False.
    xlabel : str
        x-axis label.  Defaults to name of first exchange reaction.
    ylabel : str
        y-axis label.  Defaults to name of second exchange reaction.

    Examples
    --------
    # Initialise model
    wt_ec = Yeast8Model(...)

    # Create ablation grid
    exch_rate_dict = {
        "r_1714": np.linspace(0, 2 * 8.45, 3),  # glucose
        "r_1654": np.linspace(0, 2 * 1.45, 3),  # ammonium
    }
    ablation_result_array = wt_ec.ablation_grid(exch_rate_dict)

    # Draw bar chart
    barchart_ablation_grid(exch_rate_dict, ablation_result_array)
    plt.show()
    """
    ablation_result_array = np.rot90(ablation_result_array)

    nrows, ncols = ablation_result_array.shape
    # Have exch_rate values in an extra column & row
    nrows += 1
    ncols += 1
    fig, ax = plt.subplots(nrows, ncols)

    # Define labels for legend
    # Debt: Assumes that all ablation_result DataFrames share the same format
    ablation_result_temp = ablation_result_array[0, 0]
    component_list = ablation_result_temp.priority_component.to_numpy().T
    # deletes 'original' priority component
    component_list = np.delete(component_list, 0)
    component_list = component_list.tolist()

    # Define axis labels
    global_xaxislabels = list(exch_rate_dict.values())[0]
    # Scale if specified
    global_yaxislabels = list(exch_rate_dict.values())[1][::-1]
    if percent_saturation:
        global_xaxislabels /= np.max(global_xaxislabels)
        global_xaxislabels *= 100
        global_yaxislabels /= np.max(global_yaxislabels)
        global_yaxislabels *= 100
    # Dummy value for corner position -- not used
    # (could be useful for debugging)
    global_xaxislabels = np.append([-1], global_xaxislabels)
    global_yaxislabels = np.append(global_yaxislabels, [-1])

    # Draw pie charts
    for row_idx, global_yaxislabel in enumerate(global_yaxislabels):
        # Left column reserved for exch rate 2 labels
        ax[row_idx, 0].set_axis_off()
        # Bottom left corner must be blank
        if row_idx == len(global_yaxislabels) - 1:
            pass
        else:
            # Print exch rate label
            ax[row_idx, 0].text(
                x=0.5,
                y=0.5,
                s=f"{global_yaxislabel:.3f}",
                ha="center",
                va="center",
            )
        for col_idx, global_xaxislabel in enumerate(global_xaxislabels):
            # Bottom row reserved for exch rate 1 labels
            if row_idx == len(global_yaxislabels) - 1:
                ax[row_idx, col_idx].set_axis_off()
                # Bottom left corner must be blank
                if col_idx == 0:
                    pass
                else:
                    # Print exch rate label
                    ax[row_idx, col_idx].text(
                        x=0.5,
                        y=0.5,
                        s=f"{global_xaxislabel:.3f}",
                        ha="center",
                        va="center",
                    )
            else:
                # Left column reserved for exch rate 2 labels
                if col_idx == 0:
                    pass
                else:
                    # Get times
                    ablation_result = ablation_result_array[row_idx, col_idx - 1]
                    ablation_times_df = ablation_result.loc[
                        ablation_result.priority_component != "original",
                        ablation_result.columns == "ablated_est_time",
                    ]
                    ablation_times = ablation_times_df.to_numpy().T[0]
                    print(
                        f"max time for [{row_idx}, {col_idx}] = {np.max(ablation_times)}"
                    )
                    # Deal with edge cases, e.g. negative values when exch rate is 0
                    try:
                        bar_positions = list(range(len(ablation_times)))
                        artists = ax[row_idx, col_idx].bar(
                            x=bar_positions,
                            height=ablation_times,
                            width=1,
                            color=default_mpl_colors,
                        )
                        # TODO: Determine `top` based on the max of all in the
                        # grid, rather than hard-coding
                        ax[row_idx, col_idx].set_ylim(bottom=0, top=2.7)
                        # Minimalist plot
                        ax[row_idx, col_idx].spines["top"].set_visible(False)
                        ax[row_idx, col_idx].spines["right"].set_visible(False)
                        ax[row_idx, col_idx].spines["bottom"].set_visible(False)
                        ax[row_idx, col_idx].get_xaxis().set_ticks([])
                    except:
                        print(f"Unable to draw bar chart at [{row_idx}, {col_idx}].")
                        ax[row_idx, col_idx].set_axis_off()
                        ax[row_idx, col_idx].text(
                            x=0.5,
                            y=0.5,
                            s="N/A",
                            ha="center",
                            va="center",
                            fontweight="bold",
                        )

    # For global axis labels: create a big subplot and hide everything except
    # for the labels
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    if xlabel is None:
        xlabel = list(exch_rate_dict.keys())[0]
    if ylabel is None:
        ylabel = list(exch_rate_dict.keys())[1]
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # Legend: colour = biomass component
    handles = []
    # assumes that default_mpl_colors (usually 10 elements) is longer than
    # component_list (usually 7 elements)
    # Using colour patches ref:
    # https://matplotlib.org/stable/tutorials/intermediate/legend_guide.html#creating-artists-specifically-for-adding-to-the-legend-aka-proxy-artists
    for color, component in zip(
        default_mpl_colors[: len(component_list)], component_list
    ):
        color_patch = mpatches.Patch(color=color, label=component)
        handles.append(color_patch)
    fig.legend(handles, component_list, loc="center right")
    fig.subplots_adjust(right=0.75)
