"""
Contamination detection module for X-ray group analysis.

Detects and flags potential contamination from projected low-z groups
affecting high-z group measurements.
"""

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Default cosmology
cosmo = FlatLambdaCDM(H0=67.4, Om0=0.315)


def check_projected_contamination(
    ra: np.ndarray,
    dec: np.ndarray,
    redshift: np.ndarray,
    is_detected: Optional[np.ndarray] = None,
    m200: Optional[np.ndarray] = None,
    r500: Optional[np.ndarray] = None,
    background_annulus: Optional[np.ndarray] = None,
    background_binned: Optional[np.ndarray] = None,
    low_z_threshold: float = 1.0,
    high_z_threshold: float = 1.5,
    contamination_radius_factor: float = 2.0,
    require_elevated_background: bool = True,
    background_elevation_factor: float = 1.5,
    cosmology: Optional[FlatLambdaCDM] = None
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Check for projected contamination from low-z groups affecting high-z groups.
    
    For each high-z group, checks if there are low-z groups (especially massive,
    detected ones) within the contamination radius.
    
    Parameters
    ----------
    ra : np.ndarray
        Right ascension in degrees
    dec : np.ndarray
        Declination in degrees
    redshift : np.ndarray
        Redshift array
    is_detected : np.ndarray, optional
        Boolean array indicating detected groups. If provided, only detected
        low-z groups are considered as contaminants.
    m200 : np.ndarray, optional
        M200 mass in solar masses. If provided, more massive low-z groups
        are given larger contamination radii.
    r500 : np.ndarray, optional
        R500 radius in kpc. If provided, used to define contamination radius.
    low_z_threshold : float
        Maximum redshift for "low-z" groups (default: 1.0)
    high_z_threshold : float
        Minimum redshift for "high-z" groups (default: 1.5)
    contamination_radius_factor : float
        Contamination radius = this factor × R500 (or default physical radius)
        Default: 3.0 (3×R500)
    cosmology : FlatLambdaCDM, optional
        Cosmology for distance calculations (default: Planck18)
    
    Returns
    -------
    is_contaminated : np.ndarray
        Boolean array indicating potentially contaminated high-z groups
    contamination_severity : np.ndarray
        Severity score (0-1): 0 = no contamination, 1 = severe contamination
    contamination_info : dict
        Dictionary with detailed contamination information:
        - 'contaminant_indices': List of arrays, one per group, with indices of contaminating groups
        - 'angular_separations': List of arrays with angular separations in arcsec
        - 'physical_separations': List of arrays with physical separations in kpc (at high-z group)
        - 'contaminant_masses': List of arrays with M200 of contaminants
    """
    if cosmology is None:
        cosmology = cosmo
    
    n_groups = len(ra)
    is_contaminated = np.zeros(n_groups, dtype=bool)
    contamination_severity = np.zeros(n_groups, dtype=float)
    
    # Identify low-z and high-z groups
    low_z_mask = redshift < low_z_threshold
    high_z_mask = redshift >= high_z_threshold
    
    # If is_detected provided, only consider detected low-z groups as contaminants
    if is_detected is not None:
        low_z_mask = low_z_mask & is_detected
    
    n_low_z = np.sum(low_z_mask)
    n_high_z = np.sum(high_z_mask)
    
    if n_low_z == 0 or n_high_z == 0:
        logger.info("No low-z or high-z groups found for contamination check")
        return is_contaminated, contamination_severity, {
            'contaminant_indices': [[]] * n_groups,
            'angular_separations': [[]] * n_groups,
            'physical_separations': [[]] * n_groups,
            'contaminant_masses': [[]] * n_groups
        }
    
    logger.info(f"Checking contamination: {n_low_z} low-z groups (z<{low_z_threshold}), "
                f"{n_high_z} high-z groups (z>={high_z_threshold})")
    
    # Convert to SkyCoord for efficient separation calculation
    coords = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
    
    # Get low-z group properties
    low_z_ra = ra[low_z_mask]
    low_z_dec = dec[low_z_mask]
    low_z_z = redshift[low_z_mask]
    low_z_coords = coords[low_z_mask]
    
    # Get low-z masses and radii if available
    low_z_m200 = m200[low_z_mask] if m200 is not None else None
    low_z_r500 = r500[low_z_mask] if r500 is not None else None
    
    # Get high-z group properties
    high_z_indices = np.where(high_z_mask)[0]
    high_z_ra = ra[high_z_mask]
    high_z_dec = dec[high_z_mask]
    high_z_z = redshift[high_z_mask]
    high_z_coords = coords[high_z_mask]
    
    # Initialize contamination info
    contaminant_indices = [[] for _ in range(n_groups)]
    angular_separations = [[] for _ in range(n_groups)]
    physical_separations = [[] for _ in range(n_groups)]
    contaminant_masses = [[] for _ in range(n_groups)]
    
    # For each high-z group, check for nearby low-z groups
    for i, high_idx in enumerate(high_z_indices):
        high_z_coord = high_z_coords[i]
        high_z = high_z_z[i]
        
        # Calculate angular separations to all low-z groups
        separations = high_z_coord.separation(low_z_coords)
        sep_arcsec = separations.arcsec
        
        # Determine contamination radius for each low-z group
        # If R500 available, use it; otherwise use default physical radius
        contamination_radii_arcsec = np.zeros(n_low_z)
        
        for j in range(n_low_z):
            low_z = low_z_z[j]
            
            if low_z_r500 is not None and np.isfinite(low_z_r500[j]) and low_z_r500[j] > 0:
                # Use R500-based radius (at low-z group's redshift)
                r500_kpc = low_z_r500[j]
                da_low_z = cosmology.angular_diameter_distance(low_z).to(u.kpc).value
                r500_arcsec = (r500_kpc / da_low_z) * (180 * 3600 / np.pi)
                contamination_radii_arcsec[j] = contamination_radius_factor * r500_arcsec
            else:
                # Default: use physical radius (e.g., 500 kpc) at low-z group's redshift
                default_radius_kpc = 500.0
                da_low_z = cosmology.angular_diameter_distance(low_z).to(u.kpc).value
                default_radius_arcsec = (default_radius_kpc / da_low_z) * (180 * 3600 / np.pi)
                contamination_radii_arcsec[j] = contamination_radius_factor * default_radius_arcsec
        
        # Find low-z groups within contamination radius
        within_radius = sep_arcsec < contamination_radii_arcsec
        
        if np.any(within_radius):
            # Check if background is actually elevated (real contamination)
            # vs just spatial proximity (coincidence)
            actually_contaminated = True
            
            if require_elevated_background and background_annulus is not None and background_binned is not None:
                bg_annulus_val = background_annulus[high_idx]
                bg_binned_val = background_binned[high_idx]
                
                # Check if annulus background is elevated compared to binned median
                if np.isfinite(bg_annulus_val) and np.isfinite(bg_binned_val) and bg_binned_val > 0:
                    bg_ratio = bg_annulus_val / bg_binned_val
                    # Only flag as contaminated if background is significantly elevated
                    actually_contaminated = bg_ratio > background_elevation_factor
                elif np.isfinite(bg_annulus_val) and np.isfinite(bg_binned_val):
                    # If binned is zero/negative, check if annulus is positive (contamination)
                    actually_contaminated = bg_annulus_val > 0
            
            if actually_contaminated:
                is_contaminated[high_idx] = True
            
            # Get contaminant properties
            contaminant_low_z_indices = np.where(low_z_mask)[0][within_radius]
            contaminant_seps = sep_arcsec[within_radius]
            
            # Calculate physical separations (at high-z group's redshift)
            da_high_z = cosmology.angular_diameter_distance(high_z).to(u.kpc).value
            kpc_per_arcsec_high_z = da_high_z / (180 * 3600 / np.pi)
            physical_seps = contaminant_seps * kpc_per_arcsec_high_z
            
            # Get contaminant masses
            contaminant_m200_vals = None
            if low_z_m200 is not None:
                contaminant_m200_vals = low_z_m200[within_radius]
            
            # Calculate severity score
            # Factors: number of contaminants, their masses, proximity
            n_contaminants = np.sum(within_radius)
            mass_factor = 1.0
            if contaminant_m200_vals is not None:
                # Normalize by typical group mass (10^13 M☉)
                mass_factor = np.sum(contaminant_m200_vals) / 1e13
            proximity_factor = np.sum(1.0 / (contaminant_seps + 1.0))  # Closer = worse
            
            severity = min(1.0, 0.3 * n_contaminants + 0.4 * np.log10(mass_factor + 1) + 0.3 * proximity_factor / 10.0)
            contamination_severity[high_idx] = severity
            
            # Store contamination info
            contaminant_indices[high_idx] = contaminant_low_z_indices.tolist()
            angular_separations[high_idx] = contaminant_seps.tolist()
            physical_separations[high_idx] = physical_seps.tolist()
            if contaminant_m200_vals is not None:
                contaminant_masses[high_idx] = contaminant_m200_vals.tolist()
            else:
                contaminant_masses[high_idx] = [None] * n_contaminants
    
    n_contaminated = np.sum(is_contaminated)
    logger.info(f"Found {n_contaminated} potentially contaminated high-z groups ({100*n_contaminated/n_high_z:.1f}%)")
    
    return is_contaminated, contamination_severity, {
        'contaminant_indices': contaminant_indices,
        'angular_separations': angular_separations,
        'physical_separations': physical_separations,
        'contaminant_masses': contaminant_masses
    }


def check_background_contamination(
    background_levels: np.ndarray,
    redshift: np.ndarray,
    is_detected: np.ndarray,
    contamination_mask: Optional[np.ndarray] = None,
    high_z_threshold: float = 1.5,
    contamination_factor: float = 1.5
) -> np.ndarray:
    """
    Check if background levels are elevated due to contamination.
    
    For high-z groups, flags those with background levels significantly
    higher than typical for their redshift, which may indicate contamination
    from low-z groups.
    
    Parameters
    ----------
    background_levels : np.ndarray
        Background surface brightness levels (counts/s/arcsec² or similar)
    redshift : np.ndarray
        Redshift array
    is_detected : np.ndarray
        Boolean array indicating detected groups
    contamination_mask : np.ndarray, optional
        Pre-computed contamination mask from check_projected_contamination
    high_z_threshold : float
        Minimum redshift for high-z groups (default: 1.5)
    contamination_factor : float
        Factor above median background to flag as contaminated (default: 1.5)
    
    Returns
    -------
    is_background_contaminated : np.ndarray
        Boolean array indicating groups with elevated background
    """
    high_z_mask = redshift >= high_z_threshold
    
    if not np.any(high_z_mask):
        return np.zeros_like(redshift, dtype=bool)
    
    # Calculate median background for high-z groups
    high_z_backgrounds = background_levels[high_z_mask & is_detected]
    
    if len(high_z_backgrounds) == 0:
        return np.zeros_like(redshift, dtype=bool)
    
    median_bg = np.median(high_z_backgrounds)
    threshold_bg = contamination_factor * median_bg
    
    # Flag high-z groups with elevated background
    is_background_contaminated = np.zeros_like(redshift, dtype=bool)
    is_background_contaminated[high_z_mask] = background_levels[high_z_mask] > threshold_bg
    
    # If spatial contamination mask provided, combine
    if contamination_mask is not None:
        is_background_contaminated = is_background_contaminated | contamination_mask
    
    return is_background_contaminated
