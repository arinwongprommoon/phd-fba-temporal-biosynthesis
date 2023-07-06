#!/usr/bin/env python3

import numpy as np


def get_ablation_ratio(ablation_result):
    """Get ratio to represent ablation study

    Get ratio between sum of times from ablation and longest time from
    proportional estimation, as a summary of ablation study.

    Parameters
    ----------
    ablation_result : pandas.DataFrame object
        Results of ablation study.  Columns: 'priority component' (biomass
        component being prioritised), 'ablated_flux' (flux of ablated
        biomass reaction), 'ablated_est_time' (estimated doubling time based
        on flux), 'proportional_est_time' (estimated biomass synthesis time,
        proportional to mass fraction).  Rows: 'original' (un-ablated
        biomass), other rows indicate biomass component.

    Examples
    --------
    FIXME: Add docs.

    """
    ratio, _ = _get_ablation_ratio_component(ablation_result)
    return ratio


def get_ablation_largest_component(ablation_result):
    """Get largest component from proportional estimation in ablation study


    Parameters
    ----------
    ablation_result : pandas.DataFrame object
        Results of ablation study.  Columns: 'priority component' (biomass
        component being prioritised), 'ablated_flux' (flux of ablated
        biomass reaction), 'ablated_est_time' (estimated doubling time based
        on flux), 'proportional_est_time' (estimated biomass synthesis time,
        proportional to mass fraction).  Rows: 'original' (un-ablated
        biomass), other rows indicate biomass component.

    Examples
    --------
    FIXME: Add docs.

    """
    _, largest_component = _get_ablation_ratio_component(ablation_result)
    return largest_component


@np.vectorize
def vget_ablation_ratio(ablation_result_array):
    """Get ratio to represent ablation study, apply to an array

    Get ratio between sum of times from ablation and longest time from
    proportional estimation, as a summary of ablation study.

    This is a vectorised version of get_ablation_ratio(), for convenience.

    Parameters
    ----------
    ablation_result_array : 2-dimensional numpy.ndarray of objects
        Array of ablation result DataFrames.

    Examples
    --------
    FIXME: Add docs.

    """
    return get_ablation_ratio(ablation_result_array)


@np.vectorize
def vget_ablation_largest_component(ablation_result_array):
    """Get largest component from proportional estimation, apply to an array

    Get largest component from proportional estimation in ablation study

    This is a vectorised version of get_ablation_largest_component(), for
    convenience.

    Parameters
    ----------
    ablation_result_array : 2-dimensional numpy.ndarray of objects
        Array of ablation result DataFrames.

    Examples
    --------
    FIXME: Add docs.

    """
    return get_ablation_largest_component(ablation_result_array)


def _get_ablation_ratio_component(ablation_result):
    # sum of times (ablated)
    sum_of_times = ablation_result.loc[
        ablation_result.priority_component != "original",
        ablation_result.columns == "ablated_est_time",
    ].sum()
    # get element
    sum_of_times = sum_of_times[0]

    # largest proportional_est_time, apart from original.

    # Creates reduced DataFrame that shows both priority_component and
    # proportional_est_time because I want to take note which
    # priority_component is max (in case it's not always the same).
    proportional_est_time_red = ablation_result.loc[
        ablation_result.priority_component != "original",
        ["priority_component", "proportional_est_time"],
    ]
    largest_prop_df = proportional_est_time_red.loc[
        proportional_est_time_red["proportional_est_time"].idxmax()
    ]
    largest_prop_time = largest_prop_df.proportional_est_time
    largest_prop_component = largest_prop_df.priority_component

    ratio = sum_of_times / largest_prop_time
    return ratio, largest_prop_component