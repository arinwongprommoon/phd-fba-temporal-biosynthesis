#!/usr/bin/env python3

from src.gem.yeast8model import Yeast8Model

glc_exch_rate = 16.89
wt_ec = Yeast8Model("../data/gemfiles/ecYeastGEM_batch_8-6-0.xml")
wt_ec.model.reactions.get_by_id("r_1714").bounds = (-glc_exch_rate, 0)
wt_ec.model.reactions.get_by_id("r_1714_REV").bounds = (0, glc_exch_rate)

sol = wt_ec.optimize()
orig_flux_sum = sol.fluxes.abs().sum()

breakpoint()

wt_ec.set_flux_constraint(upper_bound=900)
sol = wt_ec.optimize()
print(sol.fluxes.abs().sum())

breakpoint()

wt_ec.set_flux_constraint(upper_bound=500)
sol = wt_ec.optimize()
print(sol.fluxes.abs().sum())

breakpoint()
