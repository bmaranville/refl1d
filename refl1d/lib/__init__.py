from typing import cast, Protocol
from numpy import ndarray


class CalculationBackend(Protocol):
    @staticmethod
    def reflectivity_amplitude(
        depth: ndarray, sigma: ndarray, rho: ndarray, irho: ndarray, kz: ndarray, rho_index: ndarray, r: ndarray
    ) -> None: ...

    #    def autosampled_reflectivity_amplitude(depth, sigma, rho, irho, kz, rho_index, dR, dQ, tolerance=0.05):

    @staticmethod
    def autosampled_reflectivity_amplitude(
        depth: ndarray,
        sigma: ndarray,
        rho: ndarray,
        irho: ndarray,
        kz: ndarray,
        rho_index: ndarray,
        dR: ndarray,
        dQ: ndarray,
        tolerance: float = 0.05,
    ) -> tuple[ndarray, ndarray]:
        """
        Autosample reflectivity amplitude calculation.

        returns: (calc_kz, r)
        """
        ...
