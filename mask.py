import numpy as np
from util import Coordinate, zero_division
from field_calculator import field_distribution


def generate_mask(shape, center_coordinates=None, grating_period=(0, 0), amplitude_modulation=False,
                  amplitude_subtraction=False, beam_width=None, mode_params=None):
    center_coordinates = center_coordinates if center_coordinates else (shape[Coordinate.X] / 2,
                                                                        shape[Coordinate.Y] / 2)
    beam_width = beam_width if beam_width else min(shape) / 3
    beam_width = min(shape) * beam_width if 0 <= beam_width <= 1 else beam_width
    mode_params = mode_params if mode_params else {'mode_type': 'hermite', 'order': (0, 0)}

    x, y = np.meshgrid(np.arange(-center_coordinates[Coordinate.X],
                                 shape[Coordinate.X] - center_coordinates[Coordinate.X]),
                       np.arange(-center_coordinates[Coordinate.Y],
                                 shape[Coordinate.Y] - center_coordinates[Coordinate.Y]))

    field = field_distribution[mode_params['mode_type']](x, y, order=mode_params['order'], beam_width=beam_width)
    grating = calculate_grating(x, y, grating_period)

    amplitude_modulation = np.abs(field) if amplitude_modulation \
        else np.ones((shape[Coordinate.Y], shape[Coordinate.X]))
    amplitude_subtraction = np.abs(field) if amplitude_subtraction \
        else np.zeros((shape[Coordinate.Y], shape[Coordinate.X]))

    return amplitude_modulation * ((np.angle(field) + grating - amplitude_subtraction) % (2 * np.pi))


def calculate_grating(x, y, grating_period=(0, 0)):
    return zero_division(x * 2 * np.pi, grating_period[Coordinate.X]) \
           + zero_division(y * 2 * np.pi, grating_period[Coordinate.Y])
