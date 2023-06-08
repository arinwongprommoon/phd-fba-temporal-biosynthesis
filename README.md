# Modelling temporal partitioning of biosynthesis in yeast

Use Python version: 3.8 (tested with 3.8.14)

And then install dependencies using `poetry` <https://python-poetry.org/docs/managing-dependencies/>

If setting flux penalty (`Yeast8Model.set_flux_penalty()`), Gurobi is required as a quadratic expression of flux expressions is used.  Tested with `gurobipy` 10.01.
