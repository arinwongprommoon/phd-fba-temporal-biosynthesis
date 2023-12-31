#!/usr/bin/env python3

import numpy as np
import pickle

from scipy.spatial.distance import pdist, squareform


@np.vectorize
def vcosdist_carb_prot(ablation_result_array):
    return cosdist_carb_prot(ablation_result_array)


def cosdist_carb_prot(enz_use_array):
    distances = pdist(enz_use_array, metric="cosine")
    print(distances)
    distance_matrix = squareform(distances)
    print(distance_matrix)
    distance_triangle = np.tril(distance_matrix)
    distance_triangle[np.triu_indices(distance_triangle.shape[0])] = np.nan
    print(distance_triangle)
    metric = distances[7]
    return metric


# Alternatively, load if saved
with open("../data/interim/ec_usg_glc_amm.pkl", "rb") as handle:
    ablation_result_array = pickle.load(handle)

breakpoint()

metric = cosdist_carb_prot(ablation_result_array[0, 0])
print(metric)

breakpoint()

metric_array = np.zeros(shape=ablation_result_array.shape)

for x_index in range(ablation_result_array.shape[0]):
    for y_index in range(ablation_result_array.shape[1]):
        metric_array[x_index, y_index] = cosdist_carb_prot(
            ablation_result_array[x_index, y_index]
        )

print(metric_array)

breakpoint()

metric_array = vcosdist_carb_prot(ablation_result_array)

print(metric_array)

breakpoint()
