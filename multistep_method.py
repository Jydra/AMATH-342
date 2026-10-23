import numpy as np

from lib.constants import TOL
from lib.ode_types import real_t, np_1D_real_arr, np_2D_real_arr

from lib.ode_solver_function import LinearODEFunction
from lib.ode_solver import ODESolver, ForwardEuler, BackwardEuler
from lib.ode_solver_params import ODESolverParams


class MultistepMethod(ODESolver):
    s: int  # number of steps
    a: np_1D_real_arr  # coefficients of y
    b: np_1D_real_arr  # coefficients of f

    def __init__(
        self,
        is_explicit: bool,
        order: int,
        s: int,
        a: np_1D_real_arr,
        b: np_1D_real_arr,
    ):
        super().__init__(is_explicit, order)
        if s < 1:
            raise ValueError("Steps cannot be less than 1")
        self.s = s
        self.a = a
        self.b = b

    @classmethod
    def _cdn_a(cls, n: int):
        """Generates the conditions of a_m of Thm 2.1 as a matrix"""
        return np.array(
            [[i**j for i in range(1, n)] for j in range(1, n + 1)], dtype=real_t
        )

    @classmethod
    def _cdn_b(cls, n: int):
        """Generates the conditions of b_m in Thm 2.1 as a matrix"""
        return np.array(
            [[j * i ** (j - 1) for i in range(n)] for j in range(1, n + 1)],
            dtype=real_t,
        )

    @classmethod
    def _b_from_a(cls, a: np.ndarray, order: int, is_explicit: bool = False):
        """
        Given the coefficients a_m, solves for b_m
        """
        # TODO
        #   - Have not decided if I should keep order as a parameter
        #       > If requested order is too low, method overdefined. Accept addn'l conditions?
        #       > If requested order is too high, return an error.
        #       > Consider letting order b a driven value instead of a requested one
        if len(a) + (0 if is_explicit else 1) != order:
            raise ValueError(f"a is length {len(a)}, but order {order}")
        if np.abs(np.sum(a)) >= TOL:
            raise ValueError(f"Coefficients a must sum to 0")

        size_b = order - 1
        size_a = order - (0 if is_explicit else 1)

        A, B = MultistepMethod._cdn_a(size_a), MultistepMethod._cdn_b(size_b)

        if is_explicit:
            b = np.append(np.linalg.inv(B) @ A[:-1] @ a[1:], 0)
        else:
            b = np.linalg.inv(B) @ A @ a[1:]

        b[np.abs(b) < TOL] = 0

        return b

    @classmethod
    def _a_from_b(cls, b: np.ndarray, order: int):
        """
        Given the coefficients b_m, solves for a_m
        """

        if len(b) != order:
            raise ValueError(f"b is length {len(b)}, but order {order}")

        A, B = MultistepMethod._cdn_a(order)[:-1], MultistepMethod._cdn_b(order)

        a = np.linalg.inv(A) @ B[:-1] @ b
        a = np.insert(a, 0, -np.sum(a))

        a[np.abs(a) < TOL] = 0

        return a

    @classmethod
    def from_a(
        cls,
        a: np_1D_real_arr,
        is_explicit: bool = False,
    ):
        """
        Given rho(omega), generates a multistep method of maximal order.
        """
        if len(a) <= 1:
            raise ValueError(f"Length of a {a} must be greater than 1")
        if a[-1] == 0:
            raise ValueError(f"The value of a[-1] cannot be 0.")
        if a[-1] != 1:
            raise NotImplementedError(
                "Normalization for a, b have not yet been implemented"
            )

        s = len(a) - 1
        order = s + 1 + (0 if is_explicit else 1)
        b = MultistepMethod._b_from_a(a, order, is_explicit)
        return cls(is_explicit, order, s, a, b)

    @classmethod
    def from_b(cls, b: np_1D_real_arr):
        """
        Given sigma(omega), generates a multistep method of maximal order.
        """
        if len(b) <= 1:
            raise ValueError(f"Length of {b} must be greater than 1")

        is_explicit = False
        if np.abs(b[-1]) < TOL:
            is_explicit = True

        s = len(b) - 1
        order = s + 2

        a = MultistepMethod._a_from_b(b, order - 1)
        return cls(is_explicit, order, s, a, b)

    def set_params(self, params: ODESolverParams):
        if not self.is_explicit and isinstance(params.f, LinearODEFunction):
            I = np.eye(params.dim_y)
            asI = self.a[-1] * I
            bshA = self.b[-1] * params.h * params.f.A

            try:
                self.inv_asI_m_bshA = np.linalg.inv(asI - bshA)
            except np.linalg.LinAlgError:
                raise ValueError(
                    f"Matrix (as * I - bs * h * A) is not invertible for a: {self.a}, b:{self.b}, h:{params.h}"
                )

            bhA = params.h * self.b[:-1, None, None] * params.f.A
            aI = -self.a[:-1, None, None] * np.eye(params.dim_y)
            self.aI_p_bhA = aI + bhA

        return super().set_params(params)

    def _step(self, t, y, i):
        """
        Refer to ODESolver._step()
        """
        rel_y = y[i - self.s : i, :]
        rel_t = t[i - self.s : i]

        if self.is_explicit:
            return -np.sum(
                self.a[:-1] @ rel_y, axis=0, keepdims=False, dtype=real_t
            ) + self.params.h * self.b[:-1] @ self.params.f(rel_t, rel_y)

        if not isinstance(self.params.f, LinearODEFunction):
            return NotImplemented("step not yet implemented for implicit methods")

        # TODO How to enforce the definition of these?
        rel_y = rel_y[
            :, :, np.newaxis
        ]  # reshape rel_y to be an array of column vectors
        a = self.aI_p_bhA @ rel_y
        a = a.swapaxes(1, 2).squeeze()  # reshape back to array of row vectors
        y_rhs = np.sum(a, axis=0)
        t_rhs = np.sum(self.params.f.f_t(rel_t), axis=0)
        rhs = self.inv_asI_m_bshA @ (y_rhs + t_rhs)
        return rhs

    def _bootstrap(self, solver: ODESolver) -> np_2D_real_arr:
        """
        Uses provided bootstrapper method to generate required initial values.
        Let:
            - n be the size of the partial solution generated by the bootstrapper method
            - d be the dimension of the solution, y(T), at some time T

        Parameters:
            solver (ODESolver): the ODESolver used to generate a bootstrapped partial solution

        Returns:
            ndarray[n, d]: the partially solved solution used to bootstrap the main method
        """
        if self.s == 1:  # No bootstrap required for 1-step method
            return self.params.y_0[:, np.newaxis]

        t_star = self.params.h * (self.s - 1)
        params = ODESolverParams(
            self.params.f, self.params.h, self.params.t_0, t_star, self.params.y_0
        )
        solver.set_params(params)
        _t, y = solver.solve()
        return y

    def solve(
        self, bootstrapper: ODESolver | None = None
    ) -> tuple[np_1D_real_arr, np_2D_real_arr]:
        """
        Refer to ODESolver.solve()
        """

        # Default bootstrap method
        if bootstrapper is None:
            if self.is_explicit:
                bootstrapper = ForwardEuler()
            else:
                bootstrapper = BackwardEuler()

        y_i = self._bootstrap(bootstrapper)
        return super().solve_from_partial(y_i)

    @property
    def is_stable(self) -> bool:
        """
        Returns whether or not the MultistepMethod is stable.
        """
        # TODO
        #   - Implement
        #   - Does this depend on additional factors? Signature may have to change
        raise NotImplementedError("Stability check not yet implemented")


class Adams_Bashforth(MultistepMethod):
    def __new__(cls, steps: int):

        a = np.array([0] * (steps - 1) + [-1, 1])

        return MultistepMethod.from_a(a, is_explicit=True)


class Adams_Moulton(MultistepMethod):
    def __new__(cls, steps: int):

        a = np.array([0] * (steps - 1) + [-1, 1])

        return MultistepMethod.from_a(a, is_explicit=False)


class ExplicitNystrom(MultistepMethod):
    def __new__(cls, steps: int):
        if steps <= 1:
            raise ValueError("Explict Nystrom steps must be greater than 1")
        a = np.array([0] * (steps - 2) + [-1, 0, 1])

        return MultistepMethod.from_a(a, is_explicit=True)


class ImplicitMilne(MultistepMethod):
    def __new__(cls, steps: int):
        a = np.array([0] * (steps - 2) + [-1, 0, 1])

        return MultistepMethod.from_a(a, is_explicit=False)


class BDF(MultistepMethod):
    @classmethod
    def _beta(cls, steps: int):
        return 1.0 / np.sum(1.0 / np.arange(1, steps + 1))

    def __new__(cls, steps: int):

        beta = BDF._beta(steps)

        b = np.array([0] * steps + [beta])

        return MultistepMethod.from_b(b)
