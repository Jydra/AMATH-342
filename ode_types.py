from _collections_abc import Callable
import numpy as np

# Chosen float type
real_t = np.float64

# 1D np array of real_t
np_1D_real_arr = np.ndarray[tuple[int], np.dtype[real_t]]

# 2D np array of real_t
np_2D_real_arr = np.ndarray[tuple[int, int], np.dtype[real_t]]

# Type of function f for ODEs of form y' = f(t, y(t))
ftype = Callable[
    [
        real_t,
        np_1D_real_arr,
    ],
    np_1D_real_arr,
]

# Type of vectorized f
vftype = Callable[[np_1D_real_arr, np_2D_real_arr], np_2D_real_arr]
