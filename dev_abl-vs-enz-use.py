#!/usr/bin/env python3
#
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from yeast8model import Yeast8Model

glc_exch_rate = 16.89
wt_ec = Yeast8Model("./models/ecYeastGEM_batch_8-6-0.xml")
wt_ec.model.reactions.get_by_id("r_1714").bounds = (-glc_exch_rate, 0)
wt_ec.model.reactions.get_by_id("r_1714_REV").bounds = (0, glc_exch_rate)

wt_ec.ablation_result = wt_ec.ablate()

breakpoint()
