from abc import ABC, abstractmethod

import numba
import numpy as np
from expressions import (compile_system, program_coefs, program_step,
                         random_program_coefs, set_program_coefs)


class Equation(ABC):
    '''
    Interface of an attractor map used by createAttractors:
        name: identifier saved in the JSON files
        step: jitted map (px, py, a, b) -> (x, y)
    '''
    name: str

    @abstractmethod
    def random_coefs(self):
        pass

    @abstractmethod
    def coefs_of(self, a, b):
        pass

    @abstractmethod
    def to_json(self):
        pass

    @abstractmethod
    def from_coefs(self, a, b):
        pass


class BuiltinEquation(Equation):
    '''
    A hand-written map: "abs", "quadratic" or "trig".
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

    def coefs_of(self, a, b):
        return a, b

    def to_json(self):
        return self.name

    def from_coefs(self, a, b):
        return a, b


class DynamicEquation(Equation):
    '''Equation compiled from user strings and run by the stack machine.'''

    def __init__(self, name, x_src, y_src, coef_bounds=(-3.0, 3.0)):
        self.name = name
        self.x_src, self.y_src = x_src, y_src
        self.coef_bounds = coef_bounds
        self.template_a, self.template_b = compile_system(x_src, y_src)
        self.step = program_step

    def random_coefs(self):
        lo, hi = self.coef_bounds
        return (random_program_coefs(self.template_a, lo, hi),
                random_program_coefs(self.template_b, lo, hi))

    def coefs_of(self, a, b):
        return program_coefs(a), program_coefs(b)

    def to_json(self):
        return {"name": self.name, "x": self.x_src, "y": self.y_src}

    def from_coefs(self, a, b):
        return set_program_coefs(self.template_a, a), set_program_coefs(self.template_b, b)


def equation_from_json(spec):
    '''Build the equation saved in the "equation" field of a JSON file.'''
    if isinstance(spec, dict):
        return DynamicEquation(spec["name"], spec["x"], spec["y"])
    return BuiltinEquation(spec)

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
