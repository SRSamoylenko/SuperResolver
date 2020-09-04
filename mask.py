import numpy as np
from util import Coordinate, zero_division, ModeParams
from field_calculator import field_distribution


def generate_mask(shape, center_coordinates=None, grating_period=(0, 0), amplitude_modulation=False,
                  amplitude_subtraction=False, beam_width=None, mode_params=ModeParams()):
    if center_coordinates is None:
        center_coordinates = (shape[Coordinate.X] / 2, shape[Coordinate.Y] / 2)

    if beam_width is None:
        beam_width = min(shape) / 3

    x, y = np.meshgrid(np.arange(-center_coordinates[Coordinate.X],
                                 shape[Coordinate.X] - center_coordinates[Coordinate.X]),
                       np.arange(-center_coordinates[Coordinate.Y],
                                 shape[Coordinate.Y] - center_coordinates[Coordinate.Y]))

    field = field_distribution(x, y, beam_width=beam_width, mode_params=mode_params)
    grating = calculate_grating(x, y, grating_period)

    if not amplitude_modulation:
        amplitude_modulation = np.ones((shape[Coordinate.Y], shape[Coordinate.X]))
    else:
        amplitude_modulation = np.abs(field)

    if not amplitude_subtraction:
        amplitude_subtraction = np.zeros((shape[Coordinate.Y], shape[Coordinate.X]))
    else:
        amplitude_subtraction = np.pi * np.abs(field)

    return amplitude_modulation * ((np.angle(field) + grating - amplitude_subtraction) % (2 * np.pi))


def calculate_grating(x, y, grating_period=(0, 0)):
    return zero_division(x * 2 * np.pi, grating_period[Coordinate.X]) \
           + zero_division(y * 2 * np.pi, grating_period[Coordinate.Y])

