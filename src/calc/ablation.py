#!/usr/bin/env python3

import numpy as np

from scipy.spatial.distance import pdist
from scipy.stats import kendalltau


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
    from src.calc.ablation import get_custom_ablation_ratio
    from src.gem.yeast8model import Yeast8Model

    wt_ec = Yeast8Model("../data/gemfiles/ecYeastGEM_batch_8-6-0.xml")
    ablation_result = wt_ec.ablate()
    r = get_ablation_ratio(ablation_result)
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
    from src.calc.ablation import get_custom_ablation_ratio
    from src.gem.yeast8model import Yeast8Model

    wt_ec = Yeast8Model("../data/gemfiles/ecYeastGEM_batch_8-6-0.xml")
    ablation_result = wt_ec.ablate()
    r = get_ablation_ratio(ablation_result)
    """
    return get_ablation_ratio(ablation_result_array)


def vget_custom_ablation_ratio(ablation_result_array, component_list):
    """Get custom ratio to represent ablation study, apply to an array

    Get ratio between sum of times from ablation and longest time from
    proportional estimation, as a summary of ablation study.  The biomass
    components whose times are chosen for the calculation can be specified.

    This is a vectorised version of get_custom_ablation_ratio(), for convenience.

    Parameters
    ----------
    ablation_result_array : 2-dimensional numpy.ndarray of objects
        Array of ablation result DataFrames.
    component_list : list of str
        List of biomass components to use in sum of times.

    Examples
    --------
    from src.calc.ablation import get_custom_ablation_ratio
    from src.gem.yeast8model import Yeast8Model

    wt_ec = Yeast8Model("../data/gemfiles/ecYeastGEM_batch_8-6-0.xml")
    ablation_result = wt_ec.ablate()
    r = get_custom_ablation_ratio(ablation_result, ["protein", "carbohydrate"])
    """
    # numpy.vectorize has a 'smart' behaviour in that it tries to coerce the
    # 2nd argument to arrays.  I don't want this, hence these lines.
    # See https://stackoverflow.com/questions/4495882/numpy-vectorize-using-lists-as-arguments
    component_list_obj = np.ndarray((1,), dtype=object)
    component_list_obj[0] = component_list
    _vfunc = np.vectorize(get_custom_ablation_ratio)
    return _vfunc(ablation_result_array, component_list_obj)


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


def get_custom_ablation_ratio(ablation_result, component_list):
    sum_of_times = ablation_result.loc[
        ablation_result.priority_component.isin(component_list), ["ablated_est_time"]
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

    ratio = sum_of_times / largest_prop_time
    return ratio


def get_Tseq(ablation_result):
    """Get sequential biosynthesis time (Tseq)

    Get sequential biosythesis time (Tseq), defined as sum of times from
    ablation

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
    # sum of times (ablated)
    sum_of_times = ablation_result.loc[
        ablation_result.priority_component != "original",
        ablation_result.columns == "ablated_est_time",
    ].sum()
    # get element
    sum_of_times = sum_of_times[0]

    return sum_of_times


@np.vectorize
def vget_Tseq(ablation_result_array):
    """Get sequential biosynthesis time (Tseq), apply to an array

    Get sequential biosythesis time (Tseq), defined as sum of times from
    ablation

    This is a vectorised version of get_ablation_ratio(), for convenience.

    Parameters
    ----------
    ablation_result_array : 2-dimensional numpy.ndarray of objects
        Array of ablation result DataFrames.

    Examples
    --------
    from src.calc.ablation import get_custom_ablation_ratio
    from src.gem.yeast8model import Yeast8Model

    wt_ec = Yeast8Model("../data/gemfiles/ecYeastGEM_batch_8-6-0.xml")
    ablation_result = wt_ec.ablate()
    t = get_ablation_ratio(ablation_result)
    """
    return get_Tseq(ablation_result_array)


def get_Tseq_excl_ion(ablation_result):
    """Get sequential biosynthesis time (Tseq), excluding ions"""
    # sum of times (ablated)
    sum_of_times = ablation_result.loc[
        (ablation_result.priority_component != "original")
        & (ablation_result.priority_component != "ion"),
        ablation_result.columns == "ablated_est_time",
    ].sum()
    # get element
    sum_of_times = sum_of_times[0]

    return sum_of_times


@np.vectorize
def vget_Tseq_excl_ion(ablation_result_array):
    """Get sequential biosynthesis time (Tseq), apply to an array"""
    return get_Tseq_excl_ion(ablation_result_array)


def get_kendall_mean(enz_use_array):
    """TODO: Insert docstring"""
    distances = pdist(
        enz_use_array,
        lambda u, v: kendalltau(u, v, nan_policy="omit").statistic,
    )
    # distances[0] to distances[6] are original vs all the other biomass components
    metric = np.mean(distances[0:7])
    return metric


def get_kendall_min(enz_use_array):
    """TODO: Insert docstring"""
    distances = pdist(
        enz_use_array,
        lambda u, v: kendalltau(u, v, nan_policy="omit").statistic,
    )
    # these indices exclude original vs anything else
    # FIXME: use a 'smarter' way to do this
    metric = np.min(distances[7:28])
    return metric


@np.vectorize
def vget_kendall_mean(ablation_enzyme_flux_array):
    """TODO: Insert docstring"""
    return get_kendall_mean(ablation_enzyme_flux_array)


@np.vectorize
def vget_kendall_min(ablation_enzyme_flux_array):
    """TODO: Insert docstring"""
    return get_kendall_min(ablation_enzyme_flux_array)


def get_cosine_carb_prot(enz_use_array):
    """TODO: Insert docstring"""
    distances = pdist(enz_use_array, metric="cosine")
    # 7: distance between carbohydrate and protein
    metric = distances[13]
    return metric


@np.vectorize
def vget_cosine_carb_prot(ablation_enzyme_flux_array):
    """TODO: Insert docstring"""
    return get_cosine_carb_prot(ablation_enzyme_flux_array)
