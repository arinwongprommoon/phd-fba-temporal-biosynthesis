#!/usr/bin/env python3
import numpy as np
from src.gem.yeast8model import Yeast8Model

y = Yeast8Model("../data/gemfiles/ecYeastGEM_batch_8-6-0.xml")
abl_res = y.ablate()
for key, val in y.ablation_fluxes.items():
    print(np.min(val))

breakpoint()
