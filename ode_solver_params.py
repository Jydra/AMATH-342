from dataclasses import dataclass

from constants import TOL
from ode_types import real_t, np_1D_real_arr
from ode_function import ODEFunction


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
