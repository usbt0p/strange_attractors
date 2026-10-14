import math

import numba
import numpy as np


class Equation:
    '''
    An attractor map specification:
        name: identifier saved in the JSON files
        step: @numba.njit function (prev_x, prev_y, a, b) -> (x, y)
        coef_bounds: (low, high) bounds for the random coefficients
    '''

    def __init__(self, name: str, coef_bounds: tuple = (-3.0, 3.0)):
        self.name = name
        self.coef_bounds = coef_bounds

        match self.name:
            case "abs":
                self.step = abs_step
                self.n_coefs = 5
            case "quadratic":
                self.step = quadratic_step
                self.n_coefs = 6
            case "trig":
                self.step = trig_step
                self.n_coefs = 5
            case _:
                raise ValueError(f"Unknown equation: {self.name}")

    def random_coefs(self):
        lo, hi = self.coef_bounds
        return (np.random.uniform(lo, hi, self.n_coefs),
                np.random.uniform(lo, hi, self.n_coefs))

    def step(self, px, py, a, b): # just a wrapper so that calls are for a method
        return self.step(px, py, a, b)

# TODO will cache=True help speed??
@numba.njit(fastmath=True)
def abs_step(px, py, a, b):
    return (a[0] + a[1]*px + a[2]*py + abs(a[3]*px) + abs(a[4]*py),
            b[0] + b[1]*px + b[2]*py + abs(b[3]*px) + abs(b[4]*py))


@numba.njit(fastmath=True)
def quadratic_step(px, py, a, b):
    xx = px * px
    yy = py * py
    xy = px * py
    return (a[0] + a[1]*px + a[2]*xx + a[3]*xy + a[4]*py + a[5]*yy,
            b[0] + b[1]*px + b[2]*xx + b[3]*xy + b[4]*py + b[5]*yy)


@numba.njit(fastmath=True)
def trig_step(px, py, a, b):
    return (a[0] + a[1]*np.sin(a[2]*py) + a[3]*np.cos(a[4]*px),
            b[0] + b[1]*np.sin(b[2]*px) + b[3]*np.cos(b[4]*py))
