from abc import ABC, abstractmethod
import numpy as np

from lib.ode_types import real_t, np_1D_real_arr, np_2D_real_arr
from lib.ode_solver_params import ODESolverParams
from lib.ode_function import LinearODEFunction


# TODO
#   - Resolve: Get rid of implicit flag and make ImplicitODESolver?
#       > This way, inv(I - hA) logic for LinearODEFunction can be dealt with in __init__
#           at the most general level
class ODESolver(ABC):
    """
    Solves equations of the form y' = f(t, y(t)) by advancing forward from t_0 to
    t_0 + t_star in increments of h.
    """

    is_explicit: bool
    order: int
    params: ODESolverParams

    def __init__(self, is_explicit: bool, order: int):
        self.is_explicit = is_explicit
        self.order = order

    def set_params(self, params: ODESolverParams):
        self.params = params

    def solve_from_partial(
        self, y_i: np_2D_real_arr
    ) -> tuple[np_1D_real_arr, np_2D_real_arr]:
        """
        Continues solving ODE given a partial solution y_i
        - Required for bootstrapping in multistep methods
        - Potentially useful for situations where one solver is not universally
            preferred over the entire phase space

        Let:
            - n be the size of the given initialized solution
            - d be the dimension of the solution, y(T), at some time T

        Parameters:
            y_i (ndarray[n, d])

        Returns:
            tuple[ndarray[N], ndarray[N, d]]: (time vector t, full solution y)
        """
        if ODESolverParams is None:
            raise ValueError("Solver parameters have not been defined")

        len_yi = len(y_i)  # number of initial values given
        N = self.params.num_steps + 1  # total number of steps, incl. y_i

        y = np.zeros((N, self.params.dim_y), dtype=real_t)
        y[:len_yi, :] = y_i[:, :]
        t = np.zeros(N)
        t[:len_yi] = [self.params.h * i for i in range(len_yi)]

        for i in range(len_yi, N):
            y[i, :] = self._step(t, y, i)
            t[i] = self.params.t_0 + self.params.h * i

        return t, y

    def solve(self) -> tuple[np_1D_real_arr, np_2D_real_arr]:
        """
        Solves the ODE using the initial conditions from self.params.y_0
        """
        if self.params is None:
            raise ValueError("Solver parameters have not been defined")

        y_0 = np.array([self.params.y_0], dtype=real_t)
        t, y = self.solve_from_partial(y_0)

        return t, y

    @abstractmethod
    def _step(self, t: np_1D_real_arr, y: np_2D_real_arr, i: int) -> np_1D_real_arr:
        """
        Calculates the value of y at the next timestep.

        Let:
            - N be the total number of steps
            - d be the dimension of the solution, y(T), at some time T

        Parameters:
            t (ndarray[N]): entire time array
            y (ndarray[N, d]): entire (partially solved) solution array
            i (int): current index to be solved

        Returns:
            ndarray[d]: Newly solved data point y[i]
        """
        pass


class ForwardEuler(ODESolver):
    is_explicit = True
    order = 1

    def __init__(self):
        return

    def _step(self, t: np_1D_real_arr, y: np_2D_real_arr, i: int) -> np_1D_real_arr:
        return np.array(
            y[i - 1 : i, :]
            + self.params.h * self.params.f(t[i - 1 : i], y[i - 1 : i, :]),
            dtype=real_t,
        )


class BackwardEuler(ODESolver):
    # TODO
    #   - Implement general case. Root finding method?
    is_explicit = False
    order = 1

    def __init__(self):
        return

    def _step(self, t: np_1D_real_arr, y: np_2D_real_arr, i: int) -> np_1D_real_arr:
        if isinstance(self.params.f, LinearODEFunction):
            y_n = y[i - 1 : i, :]
            t_new = self.params.t_0 + i * self.params.h
            f_tnew = self.params.f.f_t(t_new)

            I_hA = np.eye(self.params.dim_y) - (self.params.h * self.params.f.A)

            try:
                I_hA_inv = np.linalg.inv(I_hA)
            except np.linalg.LinAlgError:
                raise ValueError(f"(I - hA) is not invertible with h={self.params.h}")

            return I_hA_inv @ (y_n + f_tnew)[0]

        raise NotImplementedError(
            "Backward Euler method not yet implemented for nonlinear f(t, y(t))"
        )
