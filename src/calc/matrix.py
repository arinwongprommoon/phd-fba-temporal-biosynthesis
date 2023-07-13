#!/usr/bin/env python3
import numpy as np


def get_susceptibility(array, x_axis, y_axis):
    """Compute susceptibility of a 2D array of values against two axes

    Parameters
    ----------
    array : 2d numpy.array
        Array of values
    x_axis : 1d numpy.array
        x axis. Must have same dimensions as array[0].
    y_axis : 1d numpy.array
        y axis. Must have same dimensions as array[1].

    Examples
    --------
    array = np.ones((10,10))
    x_axis = np.linspace(0, 1, 10)
    y_axis = np.linspace(0, 2, 10)

    sus_x, sus_y = get_susceptibility(array, x_axis, y_axis)
    """
    array_gradient = np.gradient(array, x_axis, y_axis)
    array_reciprocal = np.reciprocal(array)
    x_coeff_array = np.multiply(array_reciprocal, x_axis[np.newaxis, :])
    y_coeff_array = np.multiply(array_reciprocal, y_axis[:, np.newaxis])
    x_susceptibility = np.multiply(x_coeff_array, array_gradient[0])
    y_susceptibility = np.multiply(y_coeff_array, array_gradient[1])

    return (x_susceptibility, y_susceptibility)
