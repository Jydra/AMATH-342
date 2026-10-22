from dataclasses import dataclass
from typing import Callable
import numpy as np

from lib.constants import TOL
from lib.ode_types import real_t, np_1D_real_arr
from lib.ode_function import ODEFunction


@dataclass(frozen=True)
class ODESolverParams:
    f: ODEFunction
    h: real_t  # timestep
    t_0: real_t  # solution initial time
    t_star: real_t  # solution duration
    y_0: np_1D_real_arr  # solution initial conditions at t_0

    def __post_init__(self):
        """
        Raises ValueErrors if SolverParam is initialized with bad values
        """
        self.validate()

    def validate(self):
        if self.t_star <= 0:
            raise ValueError("t_star must be greater than 0")
        if self.h <= 0:
            raise ValueError("h must be greater than 0")

    @classmethod
    def from_raw(
        cls,
        f: Callable,
        h: real_t | float | int,
        t_0: real_t | float | int,
        t_star: real_t | float | int,
        y_0: np_1D_real_arr | list[float] | list[int] | real_t | float | int,
    ):
        """
        Creates an ODESolverParams obj from raw datatypes.
        """
        # Coerces y_0 to vector if it is passed as a scalar
        if (
            np.isscalar(y_0)
            or isinstance(y_0, real_t)  # so my linter behaves
            or isinstance(y_0, float)
            or isinstance(y_0, int)
        ):
            y_0 = np.array([y_0], dtype=real_t)

        dim = len(y_0)
        if dim == 0:
            raise ValueError(f"y_0 is length 0")

        # Coerces y_0 to np array if it is passed as a list
        if isinstance(y_0, list):
            y_0 = np.array(y_0, dtype=real_t)
        if y_0.dtype is not real_t:
            y_0 = y_0.astype(real_t)

        f = ODEFunction(dim, f)

        h = real_t(h)
        t_0 = real_t(t_0)
        t_star = real_t(t_star)

        return cls(f, h, t_0, t_star, y_0)

    @property
    def num_steps(self):
        """
        Returns the total number of timesteps of size h that discretize [t_0, t_0 + t_star]
        """
        return int((self.t_star + TOL) // self.h)

    @property
    def dim_y(self):
        """
        Returns the dimension of the solution, y(T), at some time T
        """
        return len(self.y_0)
