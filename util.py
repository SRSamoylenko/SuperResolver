from enum import Enum, IntEnum, auto
import numpy as np


class RadiationType(Enum):
    UNIFORM = auto()
    NON_UNIFORM = auto()


class SourceType(IntEnum):
    POINT = 0
    CIRCULAR = 1
    RECTANGULAR = 2
    TRIANGULAR = 3
    ROTATIONAL = 4
    POLYGONAL = 5


class Statistics(Enum):
    POISSON = auto()
    EXPONENTIAL = auto()


class SpatialModeType(Enum):
    LAGUERRE = auto()
    HERMITE = auto()


class ModeParams:
    def __init__(self, mode_type=SpatialModeType.HERMITE, n_p=0, m_l=0):
        self.mode_type = mode_type
        self.n_p = n_p
        self.m_l = m_l


class Coordinate(IntEnum):
    X = 0
    Y = 1


def cartesian_to_polar(x, y):
    return np.abs(x + y*1j), np.angle(x + y*1j)


def zero_division(a, b):
    if b == 0:
        return 0
    else:
        return a / b
