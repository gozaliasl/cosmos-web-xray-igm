"""
Spectral modeling module for X-ray sources.

Provides tools for estimating temperatures and other spectral properties
from X-ray photometry using simplified thermal models (APEC/MEKAL).
"""

import numpy as np
from scipy import optimize, interpolate
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SpectralModel:
    """Base class for X-ray spectral models."""

    def __init__(self, energy_band_kev: Tuple[float, float] = (0.5, 2.0),
                 metallicity: float = 0.3, nh_1e22: float = 0.01):
        self.energy_band = energy_band_kev
        self.metallicity = metallicity  # Solar units
        self.nh = nh_1e22 * 1e22  # Convert to cm^-2

    def count_rate_to_flux(self, temperature_kev: float) -> float:
        """
        Calculate conversion from count rate to flux for given temperature.

        This would normally use instrument response, but we use
        approximate analytical models.

        Parameters
        ----------
        temperature_kev : float
            Temperature in keV

        Returns
        -------
        conversion_factor : float
            Factor to convert counts/s to erg/cm^2/s
        """
        # Simplified conversion - in reality this requires full spectral model
        # Typical value for 0.5-2.0 keV with XMM/Chandra
        # Varies with temperature: hotter sources have harder spectra

        # Approximate scaling with temperature
        # For kT ~ 1 keV with XMM MOS: ~ 2.6e-12 erg/cm^2/s per count/s
        # Scales roughly as T^0.5
        base_factor = 2.6e-12
        temp_scaling = (temperature_kev / 1.0)**0.5

        return base_factor * temp_scaling


class APECModel(SpectralModel):
    """
    APEC thermal plasma model.

    Simplified version for temperature estimation from count rates.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._init_temperature_grid()

    def _init_temperature_grid(self):
        """Initialize temperature grid and emission model."""
        # Temperature grid in keV
        self.temp_grid = np.array([0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0])

        # Approximate emission in 0.5-2.0 keV band as function of T
        # Normalized to 1 at 1 keV
        # This is a simplified model - real APEC is more complex
        self.emission_grid = self._calculate_emission_grid()

    def _calculate_emission_grid(self) -> np.ndarray:
        """
        Calculate emission in energy band for each temperature.

        This is a simplified approximation of APEC model.
        Real implementation would use pyatomdb or XSPEC.
        """
        e_low, e_high = self.energy_band
        emission = np.zeros_like(self.temp_grid)

        for i, T in enumerate(self.temp_grid):
            # Approximate thermal bremsstrahlung + line emission
            # For low T (< 1 keV): dominated by lines, emission peaks around kT
            # For high T (> 2 keV): bremsstrahlung, emission ∝ T^0.5

            if T < 1.0:
                # Line-dominated regime
                # Peak emission around kT, with width ~ 0.3 kT
                e_peak = T
                sigma = 0.3 * T
                # Gaussian approximation for line emission
                emission[i] = np.exp(-0.5 * ((e_low - e_peak) / sigma)**2) * (e_high - e_low)
            else:
                # Bremsstrahlung-dominated
                # Emission ∝ T^0.5 × exp(-E/kT) integrated over band
                emission[i] = T**0.5 * (np.exp(-e_low/T) - np.exp(-e_high/T))

        # Normalize to 1 at 1 keV
        emission = emission / np.interp(1.0, self.temp_grid, emission)

        return emission

    def estimate_temperature(
        self,
        flux: np.ndarray,
        luminosity: np.ndarray,
        redshift: np.ndarray,
        initial_guess: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate temperature from flux and luminosity.

        Uses the fact that flux ratio in different bands depends on temperature.
        With single band, we use L-T relation as constraint.

        Parameters
        ----------
        flux : np.ndarray
            Observed flux in erg/cm^2/s
        luminosity : np.ndarray
            X-ray luminosity in erg/s
        redshift : np.ndarray
            Redshift array
        initial_guess : float
            Initial temperature guess in keV

        Returns
        -------
        temperature : np.ndarray
            Estimated temperature in keV
        temperature_error : np.ndarray
            Error on temperature
        """
        n_sources = len(flux)
        temperature = np.zeros(n_sources)
        temperature_error = np.zeros(n_sources)

        # Create interpolation function
        emissivity_func = interpolate.interp1d(
            self.temp_grid, self.emission_grid,
            kind='cubic', fill_value='extrapolate'
        )

        for i in range(n_sources):
            if flux[i] <= 0 or luminosity[i] <= 0:
                temperature[i] = np.nan
                temperature_error[i] = np.nan
                continue

            # Use L-T scaling relation as prior
            # L ∝ T^2 for groups (approximate)
            lx_44 = luminosity[i] / 1e44
            T_prior = (lx_44 ** 0.4) * 1.0  # Rough L-T relation

            # For single-band photometry, temperature is degenerate
            # Use prior from L-T relation with uncertainty
            temperature[i] = T_prior
            temperature_error[i] = 0.5 * T_prior  # ~50% uncertainty

        return temperature, temperature_error

    def predict_counts(
        self,
        temperature: float,
        emission_measure: float,
        redshift: float,
        exposure_time: float = 1.0
    ) -> float:
        """
        Predict X-ray counts for given temperature and emission measure.

        Parameters
        ----------
        temperature : float
            Temperature in keV
        emission_measure : float
            Emission measure in cm^-3
        redshift : float
            Redshift
        exposure_time : float
            Exposure time in seconds

        Returns
        -------
        predicted_counts : float
            Predicted count rate
        """
        # Get emissivity at this temperature
        emissivity_func = interpolate.interp1d(
            self.temp_grid, self.emission_grid,
            kind='cubic', fill_value='extrapolate'
        )
        emissivity = emissivity_func(temperature)

        # Luminosity = emissivity × EM × cooling_function(T)
        # Simplified: L ∝ emissivity × EM
        luminosity = emissivity * emission_measure * 1e-23  # Normalization

        # Convert to flux
        from astropy.cosmology import Planck18
        cosmo = Planck18
        lum_dist = cosmo.luminosity_distance(redshift).to('cm').value
        flux = luminosity / (4 * np.pi * lum_dist**2)

        # Convert to count rate
        conversion = self.count_rate_to_flux(temperature)
        count_rate = flux / conversion

        # Counts = count_rate × exposure
        counts = count_rate * exposure_time

        return counts


def fit_temperature_from_hardness(
    soft_counts: np.ndarray,
    hard_counts: np.ndarray,
    soft_band: Tuple[float, float] = (0.5, 1.0),
    hard_band: Tuple[float, float] = (1.0, 2.0),
    model: str = 'apec'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate temperature from hardness ratio.

    Hardness ratio = (H - S) / (H + S)

    Parameters
    ----------
    soft_counts : np.ndarray
        Counts in soft band
    hard_counts : np.ndarray
        Counts in hard band
    soft_band : tuple
        Soft energy band in keV
    hard_band : tuple
        Hard energy band in keV
    model : str
        Spectral model to use

    Returns
    -------
    temperature : np.ndarray
        Temperature in keV
    temperature_error : np.ndarray
        Error on temperature
    """
    # Calculate hardness ratio
    total_counts = soft_counts + hard_counts
    hr = (hard_counts - soft_counts) / np.maximum(total_counts, 1e-10)

    # Error on hardness ratio
    hr_error = 2 * np.sqrt(soft_counts + hard_counts) / np.maximum(total_counts, 1e-10)

    # Convert hardness ratio to temperature using model
    # This is a lookup table approach
    temp_grid = np.logspace(-0.5, 1.0, 20)  # 0.3 to 10 keV
    hr_model = np.zeros_like(temp_grid)

    for i, T in enumerate(temp_grid):
        # Calculate expected hardness ratio for this temperature
        # Simplified: use approximate spectral shape
        soft_flux = T**0.5 * (np.exp(-soft_band[0]/T) - np.exp(-soft_band[1]/T))
        hard_flux = T**0.5 * (np.exp(-hard_band[0]/T) - np.exp(-hard_band[1]/T))
        hr_model[i] = (hard_flux - soft_flux) / (hard_flux + soft_flux)

    # Interpolate to get temperature from observed hardness ratio
    hr_to_temp = interpolate.interp1d(
        hr_model, temp_grid, kind='cubic',
        bounds_error=False, fill_value=(temp_grid[0], temp_grid[-1])
    )

    temperature = hr_to_temp(hr)
    # Propagate error (approximate)
    temperature_error = np.abs(temperature * hr_error / (hr + 1e-10))

    return temperature, temperature_error


def estimate_temperature_from_luminosity_redshift(
    luminosity: np.ndarray,
    redshift: np.ndarray,
    scaling_relation: str = 'kettula2015',
    luminosity_error: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate temperature using L-T-z scaling relation.

    This uses empirical scaling relations from cluster/group studies.

    Parameters
    ----------
    luminosity : np.ndarray
        X-ray luminosity in erg/s (0.5-2.0 keV)
    redshift : np.ndarray
        Redshift
    scaling_relation : str
        Which relation to use: 'self_similar', 'mantz2016', 'pratt2009', 'kettula2015'
    luminosity_error : np.ndarray, optional
        Measurement error on luminosity. If provided, temperature error will be
        propagated from luminosity error. If None, uses intrinsic scatter.

    Returns
    -------
    temperature : np.ndarray
        Temperature in keV
    temperature_error : np.ndarray
        Error on temperature. If luminosity_error is provided, this is the
        propagated measurement error. Otherwise, it's the intrinsic scatter.
    """
    # Convert to units used in relations
    lx_44 = luminosity / 1e44  # 10^44 erg/s

    if scaling_relation == 'self_similar':
        # Self-similar scaling: L ∝ T^2 E(z)
        # Invert: T ∝ L^0.5 / E(z)^0.5
        # T ∝ L^0.5, so dT/T = 0.5 * dL/L
        from astropy.cosmology import Planck18
        cosmo = Planck18
        E_z = cosmo.H(redshift) / cosmo.H(0)

        log_T = 0.5 * np.log10(np.maximum(lx_44, 1e-10)) - 0.5 * np.log10(E_z)
        temperature = 10**log_T

        # Error propagation
        if luminosity_error is not None:
            with np.errstate(divide='ignore', invalid='ignore'):
                fractional_lum_error = luminosity_error / luminosity
                fractional_lum_error = np.where(
                    np.isfinite(fractional_lum_error) & (fractional_lum_error > 0),
                    fractional_lum_error,
                    0.0
                )
            temperature_error = temperature * 0.5 * fractional_lum_error
        else:
            # Intrinsic scatter ~ 0.3 dex
            scatter = 0.3
            temperature_error = temperature * scatter * np.log(10)

    elif scaling_relation == 'mantz2016':
        # Mantz et al. 2016 relation for groups/clusters
        # L_X = A × (T/5keV)^B × E(z)^gamma
        # With B ~ 2.5-3.0
        # T ∝ L^(1/B), so dT/T = (1/B) * dL/L
        B = 2.7
        gamma = 1.5

        from astropy.cosmology import Planck18
        cosmo = Planck18
        E_z = cosmo.H(redshift) / cosmo.H(0)

        log_T = (np.log10(np.maximum(lx_44, 1e-10)) - gamma * np.log10(E_z)) / B
        temperature = 5.0 * 10**log_T  # Pivot at 5 keV

        # Error propagation
        if luminosity_error is not None:
            with np.errstate(divide='ignore', invalid='ignore'):
                fractional_lum_error = luminosity_error / luminosity
                fractional_lum_error = np.where(
                    np.isfinite(fractional_lum_error) & (fractional_lum_error > 0),
                    fractional_lum_error,
                    0.0
                )
            temperature_error = temperature * (1.0 / B) * fractional_lum_error
        else:
            scatter = 0.2
            temperature_error = temperature * scatter * np.log(10)
    
    elif scaling_relation == 'kettula2015':
        # Kettula et al. (2015) bias-corrected Lx-Tx relation
        # log10(Lx * E(z)^-1 / Lx0) = alpha * log10(Tx / T0) + log10(A)
        # Rearranging: Tx = T0 * ((Lx * E(z)^-1 / Lx0) / A)^(1/alpha)
        #
        # Parameters from Kettula et al. (2015):
        # alpha = 2.52 (slope, bias-corrected)
        # A = 1.51 (normalization, 10^0.18)
        # Lx0 = 10^44 erg/s (pivot luminosity)
        # T0 = 5.0 keV (pivot temperature)
        # Note: Original calibration used WMAP5 (h=0.72), but we use Planck18
        alpha = 2.52  # Slope
        A = 1.51  # Normalization (10^0.18)
        Lx0 = 1e44  # Pivot luminosity in erg/s
        T0 = 5.0  # Pivot temperature in keV
        
        from astropy.cosmology import Planck18
        cosmo = Planck18
        E_z = cosmo.H(redshift) / cosmo.H(0)
        
        # Apply relation: Tx = T0 * ((Lx * E(z)^-1 / Lx0) / A)^(1/alpha)
        # Note: Lx is already in erg/s, no h-scaling needed for Planck cosmology
        Lx_over_Ez = luminosity / E_z  # Lx * E(z)^-1
        Lx_scaled = Lx_over_Ez / Lx0  # Lx * E(z)^-1 / Lx0
        
        # Handle negative or zero values
        Lx_scaled = np.maximum(Lx_scaled, 1e-10)
        temperature = T0 * (Lx_scaled / A)**(1.0 / alpha)
        
        # Error propagation: T ∝ L^(1/alpha), so dT/T = (1/alpha) * dL/L
        if luminosity_error is not None:
            with np.errstate(divide='ignore', invalid='ignore'):
                fractional_lum_error = luminosity_error / luminosity
                fractional_lum_error = np.where(
                    np.isfinite(fractional_lum_error) & (fractional_lum_error > 0),
                    fractional_lum_error,
                    0.0
                )
            temperature_error = temperature * (1.0 / alpha) * fractional_lum_error
        else:
            # Scatter in relation ~ 0.2-0.25 dex (typical for L-T relations)
            scatter = 0.22
            temperature_error = temperature * scatter * np.log(10)

    elif scaling_relation == 'pratt2009':
        # Pratt et al. 2009
        # T ∝ L^(1/2.5), so dT/T = (1/2.5) * dL/L
        log_T = (np.log10(np.maximum(lx_44, 1e-10)) + 0.3) / 2.5
        temperature = 10**log_T

        # Error propagation
        if luminosity_error is not None:
            with np.errstate(divide='ignore', invalid='ignore'):
                fractional_lum_error = luminosity_error / luminosity
                fractional_lum_error = np.where(
                    np.isfinite(fractional_lum_error) & (fractional_lum_error > 0),
                    fractional_lum_error,
                    0.0
                )
            temperature_error = temperature * (1.0 / 2.5) * fractional_lum_error
        else:
            scatter = 0.25
            temperature_error = temperature * scatter * np.log(10)
    
    else:
        raise ValueError(f"Unknown scaling relation: {scaling_relation}")

    return temperature, temperature_error


def calculate_cooling_time(
    temperature: np.ndarray,
    density: np.ndarray,
    metallicity: float = 0.3
) -> np.ndarray:
    """
    Calculate cooling time for X-ray gas.

    t_cool = (3/2) n kT / (n^2 Λ(T))

    Parameters
    ----------
    temperature : np.ndarray
        Temperature in keV
    density : np.ndarray
        Electron density in cm^-3
    metallicity : float
        Metallicity in solar units

    Returns
    -------
    cooling_time : np.ndarray
        Cooling time in years
    """
    # Temperature in Kelvin
    T_kelvin = temperature * 1.16e7

    # Cooling function Λ(T) - simplified approximation
    # For T ~ 10^6-10^7 K: Λ ~ 10^-22 erg cm^3 s^-1
    # Scales roughly as T^0.5 for bremsstrahlung
    lambda_T = 1e-22 * (T_kelvin / 1e7)**0.5

    # Thermal energy per particle
    from astropy import constants as const
    k_B = const.k_B.cgs.value
    thermal_energy = 1.5 * density * k_B * T_kelvin

    # Cooling rate per volume
    cooling_rate = density**2 * lambda_T

    # Cooling time
    t_cool_seconds = thermal_energy / cooling_rate

    # Convert to years
    seconds_per_year = 3.154e7
    t_cool_years = t_cool_seconds / seconds_per_year

    return t_cool_years
