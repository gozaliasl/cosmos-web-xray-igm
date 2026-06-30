"""
X-ray peak finding utilities for group center refinement.

Finds X-ray emission peaks near catalog group centers to correct
for offsets between catalog positions and actual X-ray centroids.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.ndimage import label, find_objects
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def find_xray_peak(
    data: np.ndarray,
    initial_x: float,
    initial_y: float,
    search_radius_pix: float = 50.0,
    smoothing_sigma: float = 2.0,
    min_snr: float = 2.0,
    error_map: Optional[np.ndarray] = None
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Find the X-ray emission peak near an initial position.
    
    Parameters
    ----------
    data : np.ndarray
        2D X-ray map data
    initial_x, initial_y : float
        Initial pixel coordinates (catalog position)
    search_radius_pix : float
        Maximum search radius in pixels (default: 50 pix ≈ 200 arcsec)
    smoothing_sigma : float
        Gaussian smoothing sigma for peak finding (default: 2.0 pixels)
    min_snr : float
        Minimum SNR for valid peak (default: 2.0)
    error_map : np.ndarray, optional
        Error map for SNR calculation. If None, uses data std as proxy.
    
    Returns
    -------
    peak_x, peak_y : float or None
        Peak pixel coordinates, or None if no valid peak found
    peak_snr : float or None
        Peak SNR, or None if no valid peak found
    """
    # Ensure initial position is within bounds
    ny, nx = data.shape
    initial_x = np.clip(initial_x, 0, nx - 1)
    initial_y = np.clip(initial_y, 0, ny - 1)
    
    # Smooth data for peak finding
    smoothed = gaussian_filter(data, sigma=smoothing_sigma)
    
    # Create search region mask
    y_coords, x_coords = np.ogrid[:ny, :nx]
    dist_sq = (x_coords - initial_x)**2 + (y_coords - initial_y)**2
    search_mask = dist_sq <= search_radius_pix**2
    
    if not np.any(search_mask):
        logger.warning(f"Search radius {search_radius_pix:.1f} pix too small or position out of bounds")
        return None, None, None
    
    # Find local maxima in search region
    # Use maximum filter to find peaks
    neighborhood_size = int(2 * smoothing_sigma) + 1
    local_max = maximum_filter(smoothed, size=neighborhood_size) == smoothed
    
    # Restrict to search region and positive values
    candidate_mask = search_mask & local_max & (smoothed > 0) & np.isfinite(smoothed)
    
    if not np.any(candidate_mask):
        logger.debug("No positive local maxima found in search region")
        return None, None, None
    
    # Calculate SNR for candidates
    if error_map is not None:
        snr_map = smoothed / (error_map + 1e-10)
    else:
        # Use local std as proxy for error
        local_std = np.nanstd(smoothed[search_mask])
        if local_std > 0:
            snr_map = smoothed / local_std
        else:
            snr_map = np.zeros_like(smoothed)
    
    candidate_snr = snr_map[candidate_mask]
    candidate_values = smoothed[candidate_mask]
    candidate_y, candidate_x = np.where(candidate_mask)
    
    # Filter by minimum SNR
    valid = candidate_snr >= min_snr
    if not np.any(valid):
        logger.debug(f"No peaks with SNR >= {min_snr:.1f} found")
        return None, None, None
    
    valid_snr = candidate_snr[valid]
    valid_values = candidate_values[valid]
    valid_x = candidate_x[valid]
    valid_y = candidate_y[valid]
    
    # Select peak with highest SNR (or highest value if SNR tied)
    best_idx = np.lexsort((valid_values, -valid_snr))[0]
    
    peak_x = float(valid_x[best_idx])
    peak_y = float(valid_y[best_idx])
    peak_snr = float(valid_snr[best_idx])
    
    # Convert to world coordinates if needed (for now return pixel coords)
    offset_pix = np.sqrt((peak_x - initial_x)**2 + (peak_y - initial_y)**2)
    offset_arcsec = offset_pix * 4.0  # Assuming 4 arcsec/pixel
    
    logger.info(f"Found X-ray peak: offset = {offset_pix:.1f} pix ({offset_arcsec:.1f} arcsec), SNR = {peak_snr:.2f}")
    
    return peak_x, peak_y, peak_snr


def find_xray_centroid(
    data: np.ndarray,
    initial_x: float,
    initial_y: float,
    aperture_radius_pix: float = 30.0,
    smoothing_sigma: float = 2.0,
    min_snr: float = 2.0,
    error_map: Optional[np.ndarray] = None
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Find X-ray emission centroid (flux-weighted center) near initial position.
    
    More robust than peak finding for extended sources.
    
    Parameters
    ----------
    data : np.ndarray
        2D X-ray map data
    initial_x, initial_y : float
        Initial pixel coordinates (catalog position)
    aperture_radius_pix : float
        Aperture radius for centroid calculation (default: 30 pix ≈ 120 arcsec)
    smoothing_sigma : float
        Gaussian smoothing sigma (default: 2.0 pixels)
    min_snr : float
        Minimum SNR for valid centroid (default: 2.0)
    error_map : np.ndarray, optional
        Error map for SNR calculation
    
    Returns
    -------
    centroid_x, centroid_y : float or None
        Centroid pixel coordinates, or None if invalid
    centroid_snr : float or None
        Mean SNR in aperture, or None if invalid
    """
    # Ensure initial position is within bounds
    ny, nx = data.shape
    initial_x = np.clip(initial_x, 0, nx - 1)
    initial_y = np.clip(initial_y, 0, ny - 1)
    
    # Smooth data
    smoothed = gaussian_filter(data, sigma=smoothing_sigma)
    
    # Create aperture mask using proper 2D coordinate arrays
    # Use meshgrid to create full 2D coordinate arrays
    y_coords, x_coords = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    
    dist_sq = (x_coords - initial_x)**2 + (y_coords - initial_y)**2
    aperture_mask = dist_sq <= aperture_radius_pix**2

    if not np.any(aperture_mask):
        return None, None, None

    # Calculate SNR
    if error_map is not None:
        snr_map = smoothed / (error_map + 1e-10)
    else:
        local_std = np.nanstd(smoothed[aperture_mask])
        if local_std > 0:
            snr_map = smoothed / local_std
        else:
            snr_map = np.zeros_like(smoothed)

    # Use only positive, finite values for centroid
    weight_map = smoothed.copy()
    weight_map[weight_map < 0] = 0
    weight_map[~np.isfinite(weight_map)] = 0

    total_weight = np.sum(weight_map[aperture_mask])
    if total_weight <= 0:
        return None, None, None

    # Calculate flux-weighted centroid
    centroid_x = np.sum(x_coords[aperture_mask] * weight_map[aperture_mask]) / total_weight
    centroid_y = np.sum(y_coords[aperture_mask] * weight_map[aperture_mask]) / total_weight
    
    # Mean SNR in aperture
    mean_snr = np.nanmean(snr_map[aperture_mask])
    
    if mean_snr < min_snr:
        logger.debug(f"Centroid SNR {mean_snr:.2f} below threshold {min_snr:.1f}")
        return None, None, None
    
    offset_pix = np.sqrt((centroid_x - initial_x)**2 + (centroid_y - initial_y)**2)
    offset_arcsec = offset_pix * 4.0  # Assuming 4 arcsec/pixel
    
    logger.info(f"Found X-ray centroid: offset = {offset_pix:.1f} pix ({offset_arcsec:.1f} arcsec), SNR = {mean_snr:.2f}")
    
    return float(centroid_x), float(centroid_y), float(mean_snr)
