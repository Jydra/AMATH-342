from inspect import signature
from abc import ABC
from typing import get_type_hints, Callable
import numpy as np

from lib.ode_types import (
    real_t,
    np_1D_real_arr,
    np_2D_real_arr,
    f_type,
    vf_type,
    ft_type,
    vft_type,
)


class ODEFunction(ABC):
    """
    Checks ftype signature, vectorized function application
    """

    dim: int
    f: f_type
    vf: vf_type  # vectorized f

    @classmethod
    def _validate_signature(cls, f: f_type) -> None:
        """
        Checks ftypeness of self.f signature or type hint.
            - if f is a def function, checks both names and types
            - if f is a lambda, checks parameter names only
        """
        if not callable(f):
            return False

        try:
            sig = signature(f)
        except (ValueError, TypeError):
            raise ValueError(f"Could not retrieve the function signature of f")

        # Example functions with correct signature.
        # TODO
        #   - Resolve: Make these functions class variables to avoid redefining?
        #       > Likely better to keep them here to enforce immutability
        #       > This should not get called often, so overhead is not that important
        def target(t: real_t, y: np_1D_real_arr) -> np_1D_real_arr: ...
        def fallback(t, y): ...

        sig_target = signature(target)
        sig_fallback = signature(fallback)

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

    def __init__(self, dim: int, f: f_type):
        self.dim = dim
        self._validate_signature(f)
        self.f = f
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


class ODETimeFunction(ABC):
    """
    Checks signature, vectorized function application
    """

    ft: ft_type
    vft: vft_type  # vectorized f

    @classmethod
    def _validate_signature(cls, f: ft_type) -> None:
        """
        Checks fttypeness of self.f signature or type hint.
            - if f_t is a def function, checks both name and type
            - if f_t is a lambda, checks parameter name only
        """
        if not callable(f):
            return False

        try:
            sig = signature(f)
        except (ValueError, TypeError):
            raise ValueError(f"Could not retrieve the function signature of f_t")

        # Example functions with correct signature.
        # TODO
        #   - Resolve: Abstract ODEFunction and this into a generic function checker/vectorizer?
        #   - Resolve: Make these functions class variables to avoid redefining?
        #       > Likely better to keep them here to enforce immutability
        #       > This should not get called often, so overhead is not that important
        def target(t: real_t) -> np_1D_real_arr: ...
        def fallback(t): ...

        sig_target = signature(target)
        sig_fallback = signature(fallback)

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

    def __init__(self, f: ft_type):
        self._validate_signature(f)
        self.f = f

        self.vf = lambda t: np.array([self.f(t_i) for i, t_i in enumerate(t)])

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
        return self.vf(t_slice)


class LinearODEFunction(ODEFunction):

    A: np_2D_real_arr
    f_t: Callable
    # I_hA_inv: np_2D_real_arr
    # h: real_t

    def __init__(
        self, A: np_2D_real_arr | real_t | int | float, f_t: Callable | None = None
    ):
        # TODO
        #   - Create ODETimeFunction, type check f_t and create vectorized __call__()
        #   - Determine generality of I_hA_inv, find cleaner way to apply
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
            self.f_t = lambda t: np.zeros((1, dim), dtype=real_t)
        else:
            self.f_t = f_t

    # def set_h(self, h: int | float | real_t) -> None:
    #     if not isinstance(h, real_t):
    #         h = real_t(h)

    #     try:
    #         self.I_hA_inv = np.linalg.inv(np.eye(self.dim) - h * self.A)
    #     except np.linalg.LinAlgError:
    #         print("Matrix (I-hA) is not invertible for given h: {h}")

    #     self.h = h
