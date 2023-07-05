#!/usr/bin/env python3


def compare_ablation_times(ablation_result1, ablation_result2, ax):
    """Compare two ablation study results

    Compare two ablation study results. Computes fold change of times (second
    study relative to the first), and draws on a bar plot on a log2 scale.

    Parameters
    ----------
    ablation_result1 : pandas.DataFrame object
        Results of ablation study.  Columns: 'priority component' (biomass
        component being prioritised), 'ablated_flux' (flux of ablated
        biomass reaction), 'ablated_est_time' (estimated doubling time based
        on flux), 'proportional_est_time' (estimated biomass synthesis time,
        proportional to mass fraction).  Rows: 'original' (un-ablated
        biomass), other rows indicate biomass component.
    ablation_result2 : pandas.DataFrame object
        Same as ablation_result1.  ablation_result2 is compared against
        ablation_result1.
    ax : matplotlib.pyplot.Axes object
        Axes to draw bar plot on.

    Examples
    --------
    # Initialise model-handling objects
    y = Yeast8Model("./models/ecYeastGEM_batch.xml")
    z = Yeast8Model("./models/ecYeastGEM_batch.xml")

    # Make z different
    z.make_auxotroph("BY4741")

    # Ablate
    y.ablation_result = y.ablate()
    z.ablation_result = z.ablate()

    # Compare ablation times
    fig, ax = plt.subplots()
    compare_ablation_times(z.ablation_result, y.ablation_result, ax)
    plt.show()
    """
    # Compute fold changes
    values_ablated1, values_proportion1 = _bar_vals_from_ablation_df(ablation_result1)
    values_ablated2, values_proportion2 = _bar_vals_from_ablation_df(ablation_result2)

    foldchange_ablated = np.array(values_ablated2) / np.array(values_ablated1)
    foldchange_proportion = np.array(values_proportion2) / np.array(values_proportion1)

    # Draw bar plot
    barwidth = 0.4
    bar_labels = ablation_result1.priority_component.to_list()
    bar_labels[0] = "all biomass"
    x_ablated = np.arange(len(bar_labels))
    x_proportion = [x + barwidth for x in x_ablated]
    ax.bar(
        x=x_ablated,
        height=np.log2(foldchange_ablated),
        width=barwidth,
        color="#3714b0",
        label="From ablating components\n in the biomass reaction",
    )
    ax.bar(
        x=x_proportion,
        height=np.log2(foldchange_proportion),
        width=barwidth,
        color="#cb0077",
        label="From mass fractions\n of each biomass component",
    )
    ax.set_xticks(
        ticks=[x + barwidth / 2 for x in range(len(x_ablated))],
        labels=bar_labels,
        rotation=45,
    )
    ax.set_xlabel("Biomass component")
    ax.set_ylabel("log2 fold change of estimated time")
    ax.legend()


def _bar_vals_from_ablation_df(ablation_result):
    """Get values for bar plots from ablation result DataFrame

    Takes DataFrame output from Yeast8Model.ablate() and reformats the data for
    use with bar plots. Specifically, computes the sum of times predicted from
    ablating the biomass reaction.

    Parameters
    ----------
    ablation_result : pandas.DataFrame object
        Results of ablation study.  Columns: 'priority component' (biomass
        component being prioritised), 'ablated_flux' (flux of ablated
        biomass reaction), 'ablated_est_time' (estimated doubling time based
        on flux), 'proportional_est_time' (estimated biomass synthesis time,
        proportional to mass fraction).  Rows: 'original' (un-ablated
        biomass), other rows indicate biomass component.

    Returns
    -------
    values_ablated : list of float
        List of estimated times predicted from ablating biomass reaction.  First
        element is the sum of times.  Subsequent elements are estimated times
        from focusing on each biomass component in turn.
    values_proportion : list of float
        List of estimated times based on proportion of estimated doubling time
        based on mass fractions of biomass components.  First element is the
        doubling time based on growth rate (i.e. flux of un-ablated biomass
        reaction).  Subsequent elements are proportional times.
    """
    # sum of times
    sum_of_times = ablation_result.loc[
        ablation_result.priority_component != "original",
        ablation_result.columns == "ablated_est_time",
    ].sum()
    # get element
    sum_of_times = sum_of_times[0]
    # ablated
    values_ablated = ablation_result.ablated_est_time.to_list()
    values_ablated[0] = sum_of_times
    # proportion
    values_proportion = ablation_result.proportional_est_time.to_list()

    return values_ablated, values_proportion
