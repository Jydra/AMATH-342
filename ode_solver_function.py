from inspect import signature
from abc import ABC, abstractmethod
from typing import get_type_hints, Callable
import numpy as np

from lib.ode_types import (
    real_t,
    np_1D_real_arr,
    np_2D_real_arr,
    f_type,
    ft_type,
)


class SolverFunction(ABC):
    f: Callable
    vf: Callable

    @classmethod
    @abstractmethod
    def _target_func_example(cls) -> Callable: ...

    @classmethod
    @abstractmethod
    def _fallback_func_example(cls) -> Callable: ...

    @property
    @abstractmethod
    def dim_in(self) -> int: ...

    @property
    @abstractmethod
    def dim_out(self) -> int: ...

    @abstractmethod  # init must be defined in child classes
    def __init__(
        self,
        f: Callable,
    ):
        self._validate_signature(f)
        self.f = f

    @classmethod
    def _validate_signature(cls, f: Callable) -> None:
        """
        Checks signature of self.f or type hint.
            - if f is a def function, checks both names and types
            - if f is a lambda, checks parameter names only
        """
        if not callable(f):
            return False

        try:
            sig = signature(f)
        except (ValueError, TypeError):
            raise ValueError(f"Could not retrieve the function signature of f")

        sig_target = signature(cls._target_func_example())
        sig_fallback = signature(cls._fallback_func_example())

        # Checks parameter types
        param_types = sig.parameters
        target_params = sig_target.parameters
        fallback_params = sig_fallback.parameters

        if not param_types:
            param_types = get_type_hints(f)

        if param_types != target_params and param_types != fallback_params:
            raise ValueError(
                f"f has parameters {param_types}; expected {target_params}"
            )

        # Checks return type
        return_type = sig.return_annotation
        target_return = sig_target.return_annotation
        fallback_return = sig_fallback.return_annotation

        if return_type != target_return and return_type != fallback_return:
            raise ValueError(
                f"f has return type {sig.return_annotation}; expected {target_return}"
            )

    @abstractmethod
    def eval(self, *args, **kwargs) -> np_1D_real_arr:
        """
        Evaluates function given arguments.

        Let:
            - d be the dimension of the solution, y(T), at some time T

        Returns:
            ndarray[d]
        """
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs) -> np_2D_real_arr:
        """
        Evaluates function for each set of args
        Requires:
            - All args are of equal length

        Let:
            - n be the number of values for which f is to be calculated
            - d be the dimension of the solution, y(T), at some time T

        Returns:
            ndarray[n, d]: The values of f
        """
        pass


class ODEFunction(SolverFunction):
    @classmethod
    def _target_func_example(cls) -> Callable:
        def target(t: real_t, y: np_1D_real_arr) -> np_1D_real_arr: ...

        return target

    @classmethod
    def _fallback_func_example(cls) -> Callable:
        def fallback(t, y): ...

        return fallback

    @property
    def dim_in(self) -> int:
        return self.dim

    @property
    def dim_out(self) -> int:
        return self.dim

    def __init__(self, dim: int, f: f_type):
        super().__init__(f)
        self.dim = dim

        test_y = np.zeros(dim, dtype=real_t)
        test_img = f(real_t(0), test_y)
        if test_img.ndim != 1:
            raise ValueError(f"The image of f is not a vector")
        if test_img.shape != (dim,):
            raise ValueError("The image of f is not the same shape as its argument")

        self.vf = lambda t, y: np.array([self.f(t_i, y[i]) for i, t_i in enumerate(t)])

    def eval(self, t: real_t, y: np_1D_real_arr) -> np_1D_real_arr:
        """
        Evaluates function at a given t, y

        Let:
            - d be the dimension of the solution, y(T), at some time T
        Parameters:
            t (real_t): The value of time, for which f(t, y(t)) is to be calculated
            y (ndarray[d]): The value of the solution, for which f(t, y(t)) is to be calculated

        Returns:
            ndarray[d]: The value of f(t, y(t))
        """
        return self.f(t, y)

    def __call__(
        self, t_slice: np_1D_real_arr, y_slice: np_2D_real_arr
    ) -> np_2D_real_arr:
        """
        Evaluates function at a slice of t, y
        Requires:
            - len(t_slice) == len(y_slice)

        Let:
            - n be the number of values for which f(t, y(t)) is to be calculated
            - d be the dimension of the solution, y(T), at some time T
        Parameters:
            t_slice (ndarray[n]): The values of time, for each of which f(t, y(t)) is to be calculated
            y_slice (ndarray[n, d]): The values of the solution, for each of which f(t, y(t)) is to be calculated

        Returns:
            ndarray[n, d]: The values of f(t, y(t))
        """
        return self.vf(t_slice, y_slice)


class ODETimeFunction(SolverFunction):
    @classmethod
    def _target_func_example(cls) -> Callable:
        def target(t: real_t) -> np_1D_real_arr: ...

        return target

    @classmethod
    def _fallback_func_example(cls) -> Callable:
        def fallback(t): ...

        return fallback

    @property
    def dim_in(self) -> int:
        return 1

    @property
    def dim_out(self) -> int:
        return self.dim

    def __init__(self, f: ft_type):
        super().__init__(f)

        test_t = real_t(0)
        test_img = f(test_t)

        if test_img.ndim != 1:
            raise ValueError(f"The image of f is not a vector")

        self.dim = test_img.shape[0]

        # Handle vectorized method with a flag?
        self.vft = lambda t: (
            lambda a: (
                np.array([self.f(t_i) for t_i in a]) if np.ndim(t) > 0 else self.f(t)
            )
        )(t)

    def eval(self, t: real_t) -> np_1D_real_arr:
        """
        Evaluates function at a given t

        Let:
            - d be the dimension of the solution, y(T), at some time T
        Parameters:
            t (real_t): The value of time, for which f_t(t) is to be calculated

        Returns:
            ndarray[d]: The value of f_t(t)
        """
        return self.f(t)

    def __call__(self, t_slice: np_1D_real_arr) -> np_2D_real_arr:
        """
        Evaluates function at a slice of t

        Let:
            - n be the number of values for which f_t(t) is to be calculated
            - d be the dimension of the solution, y(T)
        Parameters:
            t_slice (ndarray[n]): The values of time, for each of which f_t(t) is to be calculated

        Returns:
            ndarray[n, d]: The values of f_t(t)
        """
        return self.vft(t_slice)


class ODELinearFunction(ODEFunction):
    A: np_2D_real_arr
    f_t: ODETimeFunction

    def __init__(
        self,
        A: np_2D_real_arr | real_t | int | float,
        f_t: ODETimeFunction | None = None,
    ):
        if (
            np.isscalar(A)
            or isinstance(A, float)
            or isinstance(A, int)
            or isinstance(A, real_t)
        ):
            A = np.array([[A]], dtype=real_t)

        if not (A.ndim == 2):
            raise ValueError("A is not a matrix")

        if A.shape[0] != A.shape[1]:
            raise ValueError("A matrix is not square")

        dim = A.shape[0]

        if f_t is None:
            f = lambda t, y: A @ y
        else:
            f = lambda t, y: A @ y + f_t(t)

        super().__init__(dim, f)
        self.A = A

        if f_t is None:
            self.f_t = ODETimeFunction(lambda t: np.zeros(dim, dtype=real_t))
        else:
            self.f_t = f_t

    # TODO Add __call__() - there are nontrivial vectorization gains here
