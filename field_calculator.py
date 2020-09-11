import numpy as np
from util import cartesian_to_polar


def hermite_polynomial(n):
    if n == 0:
        return lambda x: x ** 0
    elif n == 1:
        return lambda x: 2 * x
    elif n >= 2:
        return lambda x: 2 * x * hermite_polynomial(n - 1)(x) - 2 * (n - 1) * hermite_polynomial(n - 2)(x)
    else:
        raise Exception("Wrong value. Value should be a positive integer.")


def laguerre_polynomial(n):
    if n == 0:
        return lambda x: x ** 0
    elif n == 1:
        return lambda x: 1 - x
    elif n >= 2:
        return lambda x: ((2 * n - 1 - x) * laguerre_polynomial(n - 1)(x)
                          - (n - 1) * laguerre_polynomial(n - 2)(x)) / n
    else:
        raise Exception("Wrong value. Value should be a positive integer.")


def generalized_laguerre_polynomial(k, alpha):
    if k == 0:
        return lambda x: x ** 0
    elif k == 1:
        return lambda x: 1 + alpha - x
    elif k >= 2:
        return lambda x: ((2 * k - 1 + alpha - x) * generalized_laguerre_polynomial(k - 1, alpha)(x)
                          - (k + alpha - 1) * generalized_laguerre_polynomial(k - 2, alpha)(x)) / k
    else:
        raise Exception("Wrong value. Value should be a positive integer.")


def generate_hermite_field(x, y, order, beam_width=1):
    return hermite_polynomial(order[0])(np.sqrt(2) * x / beam_width) \
           * hermite_polynomial(order[1])(np.sqrt(2) * y / beam_width) \
           * np.exp(-(x ** 2 + y ** 2) / beam_width ** 2)


def generate_laguerre_field(x, y, order, beam_width=1):
    r_0, phi = cartesian_to_polar(x, y)
    return (np.sqrt(2) * r_0 / beam_width) ** order[1] \
           * np.exp(- (r_0 ** 2) / beam_width ** 2) \
           * generalized_laguerre_polynomial(order[0], order[1])(2 * (r_0 ** 2) / beam_width ** 2) \
           * np.exp(-1j * order[1] * phi)


field_distribution = {
    'hermite': generate_hermite_field,
    'laguerre': generate_laguerre_field
}
