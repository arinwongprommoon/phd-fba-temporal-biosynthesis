#!/usr/bin/env python3


def get_exch_saturation(ymodel, exch_id, exch_rates, remove_glucose=True):
    """Get exchange reaction saturation curve

    Get exchange reaction saturation curve. Varies exchange reaction uptake
    value and optimises for growth at each uptake value.

    Parameters
    ----------
    ymodel : Yeast8Model object
        Model.
    exch_id : string
        Reaction ID of exchange reaction.
    exch_rates : array-like
        List of uptake values to use.
    remove_glucose : bool
        Whether to remove glucose from media.  Useful if investigating carbon
        sources.

    Examples
    --------
    y = Yeast8Model("../data/gemfiles/ecYeastGEM_batch_8-6-0.xml")
    glc_exch_rates = np.linspace(0, 18.6, 10)
    grs = get_exch_saturation(y, "r_1714", glc_exch_rates)

    import matplotlib.pyplot as plt
    plt.plot(glc_exchrates, grs)
    """
    # Check for _REV rxn
    exch_id_rev = exch_id + "_REV"
    try:
        exch_rev = ymodel.model.reactions.get_by_id(exch_id_rev)
        print(
            f"Reversible exchange reaction {exch_id_rev} found and taken into account."
        )
        exch_rev_present = True
    except KeyError as e:
        print(
            f"Error-- reversible exchange reaction {exch_id_rev} not found. Ignoring."
        )
        exch_rev_present = False

    # Kill glucose
    if remove_glucose:
        ymodel.remove_media_components(["r_1714", "r_1714_REV"])

    growthrates = []
    for exch_rate in exch_rates:
        # negative due to FBA conventions re exchange reactions
        ymodel.model.reactions.get_by_id(exch_id).bounds = (-exch_rate, 0)
        # positive due to FBA conventions re reversible reaction
        if exch_rev_present:
            exch_rev.bounds = (0, exch_rate)
        ymodel.solution = ymodel.optimize()
        growthrates.append(ymodel.solution.fluxes[ymodel.growth_id])
    return growthrates
