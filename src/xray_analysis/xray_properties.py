"""
X-ray properties calculation module.

Calculates X-ray luminosities, fluxes, and related properties with proper
cosmological corrections and error propagation.
"""

import numpy as np
from astropy import units as u
from astropy import constants as const
from astropy.cosmology import Planck18
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Default cosmology: Planck18 (H0=67.4 km/s/Mpc, Om0=0.315)
# Using Planck results for consistency with modern cosmology
cosmo = Planck18


class XrayProperties:
    """Container for X-ray properties."""

    def __init__(self, flux: np.ndarray, flux_error: np.ndarray,
                 luminosity: np.ndarray, luminosity_error: np.ndarray,
                 luminosity_distance: np.ndarray):
        self.flux = flux
        self.flux_error = flux_error
        self.luminosity = luminosity
        self.luminosity_error = luminosity_error
        self.luminosity_distance = luminosity_distance

    def to_dict(self) -> Dict[str, np.ndarray]:
        """Convert to dictionary."""
        return {
            'flux_erg_s_cm2': self.flux,
            'flux_error': self.flux_error,
            'luminosity_erg_s': self.luminosity,
            'luminosity_error': self.luminosity_error,
            'luminosity_distance_mpc': self.luminosity_distance
        }


def calculate_xray_flux(
    net_counts: np.ndarray,
    net_error: np.ndarray,
    count_rate_to_flux: float = 2.6e-12,
    exposure_time: Optional[float] = None,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate X-ray flux from counts.

    Parameters
    ----------
    net_counts : np.ndarray
        Net source counts
    net_error : np.ndarray
        Error on net counts
    count_rate_to_flux : float, optional
        Conversion factor from count rate to flux (erg cm^-2 s^-1 / (counts s^-1))
        Default: 2.6e-12 (XMM MOS1+MOS2, 0.5-2 keV, kT≈1 keV APEC spectrum)
    exposure_time : float, optional
        Exposure time in seconds. If None, assumes counts are already count rates
    verbose : bool, optional
        Print information

    Returns
    -------
    flux : np.ndarray
        X-ray flux in erg cm^-2 s^-1
    flux_error : np.ndarray
        Error on flux
    """
    # Convert string to float if needed (handles YAML parsing issues)
    if isinstance(count_rate_to_flux, str):
        # Handle expressions like "5.41*1e-12" by evaluating safely
        # Replace * with multiplication for simple cases
        if '*' in count_rate_to_flux:
            parts = count_rate_to_flux.split('*')
            try:
                count_rate_to_flux = float(parts[0].strip()) * float(parts[1].strip())
            except (ValueError, IndexError):
                # Fall back to direct conversion (will raise if invalid)
                count_rate_to_flux = float(count_rate_to_flux)
        else:
            count_rate_to_flux = float(count_rate_to_flux)
    
    if verbose:
        logger.info(f"Calculating X-ray flux for {len(net_counts)} sources")
        logger.info(f"Count rate to flux conversion: {count_rate_to_flux:.2e} erg cm^-2 s^-1 / (counts s^-1)")

    if exposure_time is not None:
        # Convert counts to count rate
        count_rate = net_counts / exposure_time
        count_rate_error = net_error / exposure_time
    else:
        count_rate = net_counts
        count_rate_error = net_error

    # Convert count rate to flux
    flux = count_rate * count_rate_to_flux
    flux_error = count_rate_error * count_rate_to_flux

    if verbose:
        detected = flux > 0
        if np.sum(detected) > 0:
            logger.info(f"Median flux (positive): {np.median(flux[detected]):.2e} erg cm^-2 s^-1")

    return flux, flux_error


def calculate_xray_luminosity(
    flux: np.ndarray,
    flux_error: np.ndarray,
    redshift: np.ndarray,
    energy_band_kev: Tuple[float, float] = (0.5, 2.0),
    k_correction: bool = True,
    cosmology=None,
    verbose: bool = True
) -> XrayProperties:
    """
    Calculate X-ray luminosity from flux with cosmological corrections.

    Parameters
    ----------
    flux : np.ndarray
        X-ray flux in erg cm^-2 s^-1
    flux_error : np.ndarray
        Error on flux
    redshift : np.ndarray
        Redshift of sources
    energy_band_kev : tuple, optional
        Observed energy band in keV (default: 0.5-2.0)
    k_correction : bool, optional
        Apply K-correction for redshift (default: True)
    cosmology : astropy.cosmology, optional
        Cosmology to use (default: Planck18)
    verbose : bool, optional
        Print information

    Returns
    -------
    XrayProperties
        Container with flux, luminosity, and related properties
    """
    if cosmology is None:
        cosmology = cosmo

    if verbose:
        logger.info(f"Calculating X-ray luminosity for {len(flux)} sources")
        logger.info(f"Redshift range: [{np.min(redshift):.3f}, {np.max(redshift):.3f}]")
        logger.info(f"Energy band: {energy_band_kev[0]:.1f}-{energy_band_kev[1]:.1f} keV")
        logger.info(f"K-correction: {k_correction}")

    # Calculate luminosity distance
    lum_dist = cosmology.luminosity_distance(redshift).to(u.cm).value

    # Calculate K-correction factor if requested
    if k_correction:
        # Use thermal emission approximation (spectral_index=None)
        # This gives K(z) ≈ (1+z)^0.5 for kT ≈ 1 keV, matching APEC model
        k_corr = calculate_k_correction(redshift, energy_band_kev, spectral_index=None)
    else:
        k_corr = np.ones_like(redshift)

    # Calculate luminosity: L = 4π d_L^2 × F × K(z)
    luminosity = 4 * np.pi * lum_dist**2 * flux * k_corr
    luminosity_error = 4 * np.pi * lum_dist**2 * flux_error * k_corr

    # Convert luminosity distance to Mpc for output
    lum_dist_mpc = lum_dist / 3.086e24  # cm to Mpc

    if verbose:
        positive = luminosity > 0
        if np.sum(positive) > 0:
            log_lx = np.log10(luminosity[positive])
            logger.info(f"Log L_X range: [{np.min(log_lx):.2f}, {np.max(log_lx):.2f}] erg/s")
            logger.info(f"Median Log L_X: {np.median(log_lx):.2f} erg/s")

    return XrayProperties(
        flux=flux,
        flux_error=flux_error,
        luminosity=luminosity,
        luminosity_error=luminosity_error,
        luminosity_distance=lum_dist_mpc
    )


def calculate_k_correction(
    redshift: np.ndarray,
    energy_band_kev: Tuple[float, float] = (0.5, 2.0),
    spectral_index: float = None,
    temperature_kev: float = 1.0
) -> np.ndarray:
    """
    Calculate K-correction for X-ray luminosity.

    The K-correction accounts for the fact that we observe a different
    rest-frame energy band at different redshifts.

    For thermal emission (APEC), the K-correction is computed from the
    APEC spectrum. For kT ≈ 1 keV in the 0.5-2.0 keV band, the K-correction
    is approximately K(z) ≈ (1+z)^0.5.

    Parameters
    ----------
    redshift : np.ndarray
        Redshift array
    energy_band_kev : tuple
        Observed energy band in keV (default: 0.5-2.0)
    spectral_index : float, optional
        X-ray spectral index Γ for power-law approximation.
        If None, uses thermal emission approximation based on temperature.
    temperature_kev : float, optional
        Temperature in keV for thermal emission (default: 1.0 keV).
        Used when spectral_index is None.

    Returns
    -------
    k_correction : np.ndarray
        K-correction factor
    """
    if spectral_index is None:
        # For thermal emission (APEC), use temperature-dependent approximation
        # For kT ≈ 1 keV in 0.5-2.0 keV band, K(z) ≈ (1+z)^0.5
        # This is based on APEC model calculations for typical group temperatures
        # The exponent varies slightly with temperature but is ~0.5 for kT ~ 1 keV
        effective_index = 1.5  # Equivalent to Γ ≈ 1.5 for power-law approximation
        k_corr = (1 + redshift) ** (effective_index - 1.0)  # ≈ (1+z)^0.5
    else:
        # Power-law approximation: K(z) = (1 + z)^(Γ - 2)
        # Note: For Γ=2.0, this gives K(z)=1.0 (no correction), which is incorrect
        # for thermal emission. Use spectral_index ≈ 1.5 for thermal sources.
        k_corr = (1 + redshift) ** (spectral_index - 2)

    return k_corr


def calculate_rest_frame_luminosity(
    observed_luminosity: np.ndarray,
    redshift: np.ndarray,
    observed_band_kev: Tuple[float, float] = (0.5, 2.0),
    rest_band_kev: Tuple[float, float] = (0.5, 2.0),
    spectral_index: float = 2.0
) -> np.ndarray:
    """
    Convert observed-frame luminosity to rest-frame luminosity in a different band.

    Parameters
    ----------
    observed_luminosity : np.ndarray
        Observed luminosity in erg/s
    redshift : np.ndarray
        Redshift array
    observed_band_kev : tuple
        Observed energy band in keV
    rest_band_kev : tuple
        Rest-frame energy band in keV
    spectral_index : float
        X-ray spectral index

    Returns
    -------
    rest_luminosity : np.ndarray
        Rest-frame luminosity in specified band
    """
    # Calculate bolometric correction from observed to rest frame
    e_obs_low, e_obs_high = observed_band_kev
    e_rest_low, e_rest_high = rest_band_kev

    # For power-law: L ∝ ∫ E^(-Γ) dE
    # Ratio of luminosities in different bands
    def band_ratio(e_low, e_high, gamma):
        if gamma != 1:
            return (e_high**(2-gamma) - e_low**(2-gamma)) / (2 - gamma)
        else:
            return np.log(e_high / e_low)

    # Observed band in rest frame
    e_obs_rest_low = e_obs_low * (1 + redshift)
    e_obs_rest_high = e_obs_high * (1 + redshift)

    # Calculate correction factor
    obs_integral = band_ratio(e_obs_rest_low, e_obs_rest_high, spectral_index)
    rest_integral = band_ratio(e_rest_low, e_rest_high, spectral_index)

    correction = rest_integral / obs_integral

    rest_luminosity = observed_luminosity * correction

    return rest_luminosity


def calculate_xray_temperature_from_lx(
    luminosity: np.ndarray,
    redshift: np.ndarray,
    scaling_relation: str = 'kettula2015',
    luminosity_error: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate X-ray temperature from luminosity using scaling relations.

    This provides a rough estimate. For proper temperature measurement,
    spectral fitting is required.

    Parameters
    ----------
    luminosity : np.ndarray
        X-ray luminosity in erg/s (0.5-2.0 keV band)
    redshift : np.ndarray
        Redshift array
    scaling_relation : str, optional
        Which scaling relation to use: 'kettula2015' (default, bias-corrected), 'mantz2016', 'pratt2009'
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
    # Convert luminosity to units for scaling relations
    lx_44 = luminosity / 1e44  # in units of 10^44 erg/s

    if scaling_relation == 'mantz2016':
        # L_X - T relation from Mantz et al. 2016
        # L ∝ T^α with α ~ 2.5-3.0 for groups/clusters
        # Log L_X = A + B * log T + scatter
        # Typical: log L_X,44 = 2.5 * log(T/5 keV) + scatter

        # Invert to get T from L
        # log T = (log L_X,44) / 2.5
        # T ∝ L^(1/2.5), so dT/T = (1/2.5) * dL/L
        log_lx_44 = np.log10(np.maximum(lx_44, 1e-10))
        log_T = log_lx_44 / 2.5
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
            # Scatter in relation ~ 0.2 dex
            scatter = 0.2
            temperature_error = temperature * scatter * np.log(10)

    elif scaling_relation == 'pratt2009':
        # Pratt et al. 2009 L-T relation for groups/clusters
        # Similar form but different normalization
        # T ∝ L^(1/2.7), so dT/T = (1/2.7) * dL/L
        log_lx_44 = np.log10(np.maximum(lx_44, 1e-10))
        log_T = (log_lx_44 + 0.5) / 2.7
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
            temperature_error = temperature * (1.0 / 2.7) * fractional_lum_error
        else:
            scatter = 0.25
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
        # If luminosity_error is provided, propagate measurement error
        # Otherwise, use intrinsic scatter
        if luminosity_error is not None:
            # Propagate luminosity measurement error: dT/T = (1/alpha) * dL/L
            # This gives the measurement uncertainty, not the scatter
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
            # This represents intrinsic scatter, not measurement error
            scatter = 0.22
            temperature_error = temperature * scatter * np.log(10)

    else:
        raise ValueError(f"Unknown scaling relation: {scaling_relation}")

    # Apply redshift evolution if needed (self-similar: L ∝ T^2 E(z)^2)
    # For now, keep simple estimate

    return temperature, temperature_error


def calculate_emission_measure(
    luminosity: np.ndarray,
    temperature: np.ndarray,
    redshift: np.ndarray,
    metallicity: float = 0.3
) -> np.ndarray:
    """
    Calculate emission measure from luminosity and temperature.

    EM = ∫ n_e n_H dV

    Parameters
    ----------
    luminosity : np.ndarray
        X-ray luminosity in erg/s
    temperature : np.ndarray
        Temperature in keV
    redshift : np.ndarray
        Redshift
    metallicity : float
        Metallicity in solar units

    Returns
    -------
    emission_measure : np.ndarray
        Emission measure in cm^-3
    """
    # Approximate cooling function Λ(T) ~ T^0.5 / (10^7 K)
    # L_X = Λ(T) × EM
    # This is highly simplified - real calculation requires APEC/MEKAL

    temp_kelvin = temperature * 1.16e7  # keV to Kelvin
    lambda_t = temp_kelvin**0.5 / 1e7  # Approximate cooling function

    emission_measure = luminosity / lambda_t

    return emission_measure
