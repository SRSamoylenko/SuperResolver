from enum import Enum, IntEnum
import numpy as np


class Coordinate(IntEnum):
    X = 0
    Y = 1


def cartesian_to_polar(x, y):
    return np.abs(x + y*1j), np.angle(x + y*1j)


def zero_division(a, b):
    return a/b if b else 0
