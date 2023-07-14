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

    return [x_susceptibility, y_susceptibility]


class ArrayCollection:
    def __init__(self, raw_array, x_axis, y_axis):
        # Store raw input
        self.raw = raw_array

        # Main use
        self.array = self.raw
        # Replace pixels that correspond to exch rate 0 with NaNs
        # These are prone to weird values
        self.array[0, :] = np.nan
        self.array[:, 0] = np.nan

        # Susceptibility
        _sus = get_susceptibility(self.array, x_axis, y_axis)
        self.sus = GradientCollection(*_sus)

        # Gradient
        _gradient = np.gradient(self.array)
        self.gradient = GradientCollection(*_gradient)

        # For streamplots
        # rot90 and flipping values to get orientation & arrow right because
        # matplotlib and seaborn use different axes directions
        # Only sus for now -- will extend to gradient when needed
        _sus_sp = get_susceptibility(np.rot90(self.array), x_axis, y_axis[::-1])
        self.sus_sp = StreamplotInputs(*_sus_sp)


class GradientCollection:
    def __init__(self, x_gradient_array, y_gradient_array):
        # Store raw input
        self.x = x_gradient_array
        self.y = y_gradient_array

        # Magnitude
        self.magnitudes = np.sqrt(self.x**2, self.y**2)

        # Magnitude of x subtracted by magnitude of y
        # Useful for seeing which area is x- or y-limiting
        self.greater = np.abs(self.x) - np.abs(self.y)


class StreamplotInputs:
    def __init__(self, x_gradient_array, y_gradient_array):
        # Store raw input
        self.x = x_gradient_array
        self.y = y_gradient_array

        # Flipping to play well with streamplot
        self.y = -self.y

        # Magnitude
        self.magnitudes = np.sqrt(self.x**2, self.y**2)
