import numpy as np

from lib.ode_types import real_t, np_1D_real_arr, np_2D_real_arr

from lib.ode_solver import ODESolver


class RungeKutta(ODESolver):
    A: np_2D_real_arr
    b: np_1D_real_arr
    c: np_1D_real_arr

    rk_dim: int

    def __init__(self, A: np_2D_real_arr, b: np_1D_real_arr, c: np_1D_real_arr):
        if A.ndim != 2:
            raise ValueError("A is not a matrix")
        rk_dim = A.shape[0]
        if A.shape[1] != rk_dim:
            raise ValueError("A is not square")
        if len(b) != rk_dim:
            raise ValueError("b is the wrong length")
        if len(c) != rk_dim:
            raise ValueError("c is the wrong length")

        self.A = A
        self.b = b
        self.c = c
        self.rk_dim = rk_dim

        implicit_indices = np.triu_indices(rk_dim, k=0)
        implicit_elements = A[implicit_indices]
        self.is_explicit = bool(np.all(implicit_elements == 0))

    def _step(self, t, y, i):
        if self.is_explicit:
            ydim = np.shape(y[0])[0]

            t_i = t[i - 1 : i]
            t_rel = t_i + self.c * self.params.h

            y_i = y[i - 1 : i, :]
            k = np.zeros((self.rk_dim, ydim), dtype=real_t)
            k[0] = y_i

            for j in range(1, self.rk_dim):
                f = self.params.f(t_rel[:j], k[:j, :])
                k[j] = y_i + self.params.h * np.sum(
                    self.A[j][:j][:, np.newaxis] * f, axis=0
                )

            f = self.params.f(t_rel, k)
            return y_i + self.params.h * np.sum(self.b[:, np.newaxis] * f, axis=0)

        raise NotImplementedError("IRK not yet implemented")

    def display_tableaux(self):
        raise NotImplementedError()


class RK4(RungeKutta):
    def __new__(cls):
        A = np.array([[0, 0, 0, 0], [0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 1, 0]])
        b = np.array([1.0 / 6, 1.0 / 3, 1.0 / 3, 1.0 / 6])
        c = np.array([0, 0.5, 0.5, 2.0 / 3])

        return RungeKutta(A, b, c)
