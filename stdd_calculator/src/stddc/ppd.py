"""Compute and analyse the phase plane diagram (PPD)."""

# /usr/bin/env python3

import copy
import warnings
from typing import Optional

import numpy as np
import numpy.typing as npt
from scipy.integrate import trapezoid
from scipy.interpolate import CubicSpline, RegularGridInterpolator
from tqdm.contrib.itertools import product

from stddc import STDDMaker
from stddc.distr import SalFunc


def flatten(lst: list) -> list:
    """Flatten a list of lists into a single list."""
    return [item for sublist in lst for item in sublist]


def reshape_2d(flat_list: list, num_rows: int) -> list:
    """Reshape a flat list into a 2D list with num_rows rows."""
    if len(flat_list) % num_rows != 0:
        raise ValueError("The length of the list must be divisible by num_rows")

    num_cols = len(flat_list) // num_rows
    return [flat_list[i * num_cols : (i + 1) * num_cols] for i in range(num_rows)]


class PPDMaker:
    """Computes SAL values over a 2D grid of synaptic weights (w_12, w_21) to form a phase plane diagram."""

    def __init__(self, stddmaker: STDDMaker, w_range: npt.NDArray):
        """
        Args:
            stddmaker: Template STDDMaker instance; copied for each grid point.
            w_range: 1D array of weight values defining the grid axes.
        """
        self.stddmaker = stddmaker
        self.w_range = w_range
        self.w_len = len(w_range)
        self.sal_func = None
        self._stddmaker_grid = []
        self._distribute_stddmakers()
        self.sal_grid = np.zeros((self.w_len, self.w_len, 2))

    @property
    def sal_12(self) -> npt.NDArray:
        """SAL values for W_12 (shape: w_len × w_len)."""
        return self.sal_grid[:, :, 0]

    @property
    def sal_21(self) -> npt.NDArray:
        """SAL values for W_21 (shape: w_len × w_len)."""
        return self.sal_grid[:, :, 1]

    def _distribute_stddmakers(self) -> None:
        """Populate _stddmaker_grid with STDDMaker copies for each (w_12, w_21) combination."""
        for w in self.w_range:
            sub_arr = []
            for v in self.w_range:
                sm = copy.copy(self.stddmaker)
                sm.w_12 = v
                sm.w_21 = w
                sub_arr.append(sm)
            self._stddmaker_grid.append(sub_arr)

    def calc_sal_grid(self, func: SalFunc, *args: int | float) -> npt.NDArray:
        """Compute SAL values at every grid point and store in self.sal_grid.

        Args:
            func: Kernel function passed to STDDMaker.calc_sal.
            *args: Additional arguments passed to func.

        Returns:
            sal_grid array of shape (w_len, w_len, 2).
        """
        for i, j in product(range(self.w_len), range(self.w_len)):
            self.sal_grid[i, j, :] = self._stddmaker_grid[i][j].calc_sal(func, *args)

        return self.sal_grid

    def upscale_sal_grid(self, n: int) -> npt.NDArray:
        """Interpolate sal_grid to a finer n×n grid using cubic splines.

        Args:
            n: Number of points along each axis in the upscaled grid.

        Returns:
            Upscaled sal_grid of shape (n, n, 2).
        """
        self.w_range_hd = np.linspace(self.w_range[0], self.w_range[-1], n)
        self.sal_grid_hd = np.empty((n, n, 2))

        sal_ip_0 = RegularGridInterpolator(
            (self.w_range, self.w_range),
            self.sal_grid[:, :, 0],
            bounds_error=False,
            fill_value=None,
        )
        sal_ip_1 = RegularGridInterpolator(
            (self.w_range, self.w_range),
            self.sal_grid[:, :, 1],
            bounds_error=False,
            fill_value=None,
        )

        ww_hd, vv_hd = np.meshgrid(
            self.w_range_hd, self.w_range_hd, indexing="ij", sparse=True
        )

        self.sal_grid_hd[:, :, 0] = sal_ip_0((ww_hd, vv_hd), method="cubic")
        self.sal_grid_hd[:, :, 1] = sal_ip_1((ww_hd, vv_hd), method="cubic")

        return self.sal_grid_hd

    @staticmethod
    def _calc_roots(
        x: npt.NDArray,
        y: npt.NDArray,
        vals: npt.NDArray,
        axis: str = "y",
        warn_mulitple: bool = True,
    ) -> npt.NDArray:
        """Find zero crossings of a 2D field along one axis using cubic splines.

        Args:
            x: Values along the x axis.
            y: Values along the y axis.
            vals: 2D array of shape (len(x), len(y)).
            axis: Axis along which to find roots ('x' or 'y').
            warn_mulitple: Warn if more than one root is found at a given slice.

        Returns:
            Array of shape (n_roots, 2) with [other_axis_value, root] pairs.
        """
        if axis == "y":
            root_axis = y
            other_axis = x
            vals = vals.T
        elif axis == "x":
            root_axis = x
            other_axis = y
        else:
            raise ValueError("'axis' must be either 'x' or 'y'.")

        all_roots = []
        for i, val in enumerate(other_axis):
            spline = CubicSpline(root_axis, vals[i], extrapolate=False)
            roots = spline.roots()

            if len(roots) > 1 and warn_mulitple:
                warnings.warn(f"Found {len(roots)} > 1 at x={val}!")

            for r in roots:
                all_roots.append([val, r])

        return np.array(all_roots)

    def calc_boa(self, upscale: Optional[int] = None, axis: str = "y") -> npt.NDArray:
        """Compute the basin of attraction curve in the PPD.

        Args:
            upscale: If given, interpolate sal_grid to an upscale×upscale grid first.
            axis: Axis along which to find the zero crossing of sal_12.

        Returns:
            Array of shape (n_roots, 2) with [w_21, w_12] pairs on the nullcline.
        """

        if upscale is None:
            self.w_range_hd = self.w_range
            self.sal_grid_hd = self.sal_grid
        else:
            self.upscale_sal_grid(upscale)

        boa = self._calc_roots(
            self.w_range_hd, self.w_range_hd, self.sal_grid_hd[:, :, 0], axis="y"
        )

        return boa

    @staticmethod
    def rotate_boa(boa: npt.NDArray) -> npt.NDArray:
        """Rotate BOA coordinates to (mean weight, asymmetry) axes."""
        w = 0.5 * (boa[:, 0] + boa[:, 1])
        d = -0.5 * (boa[:, 0] - boa[:, 1])
        return np.column_stack((w, d))

    @staticmethod
    def deviation_of_boa(boa: npt.NDArray) -> float:
        """Compute the normalized area between the BOA and the diagonal."""
        w = 0.5 * (boa[:, 0] + boa[:, 1])
        d = 0.5 * np.abs(boa[:, 0] - boa[:, 1]) / np.abs(w)
        area = trapezoid(d, x=boa[:, 0])
        area /= np.abs(boa[-1, 0] - boa[0, 0])
        return area
