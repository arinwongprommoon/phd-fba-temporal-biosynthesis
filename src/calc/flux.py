#!/usr/bin/env python3


def compare_fluxes(ymodel1, ymodel2):
    """Compare fluxes between two models

    Compare fluxes between two models.  If fluxes aren't already computed and
    stored, run simulations first.

    Parameters
    ----------
    ymodel1 : Yeast8Model object
        Model 1.
    ymodel2 : Yeast8Model object
        Model 2.

    Returns
    -------
    pandas.DataFrame object
        Fluxes, sorted by magnitude, large changes on top.  Indices show
        reaction ids, values show the fluxes (with signs, positive or negative)

    Examples
    --------
    # Initialise model-handling objects
    y = Yeast8Model("../data/gemfiles/ecYeastGEM_batch.xml")
    z = Yeast8Model("../data/gemfiles/ecYeastGEM_batch.xml")

    # Make z different
    z.make_auxotroph("BY4741")

    # Compare fluxes
    dfs = compare_fluxes(y, z)
    """
    # Check if fluxes have already been computed.
    # If not, compute automatically.
    for idx, ymodel in enumerate([ymodel1, ymodel2]):
        if ymodel.solution is None:
            print(f"Model {idx+1} has no stored solution, optimising...")
            ymodel.solution = ymodel.optimize()

    diff_fluxes = ymodel2.solution.fluxes - ymodel1.solution.fluxes
    nonzero_idx = diff_fluxes.to_numpy().nonzero()[0]
    diff_fluxes_nonzero = diff_fluxes[nonzero_idx]
    # Sort by absolute flux value, large changes on top.
    diff_fluxes_sorted = diff_fluxes_nonzero[
        diff_fluxes_nonzero.abs().sort_values(ascending=False).index
    ]

    return diff_fluxes_sorted
