"""
Aperture photometry module for X-ray sources.

Performs aperture photometry on X-ray maps with proper background subtraction
and error propagation.
"""

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class PhotometryResult:
    """Container for photometry results."""

    def __init__(self, source_counts: np.ndarray, source_error: np.ndarray,
                 background: np.ndarray, background_error: np.ndarray,
                 net_counts: np.ndarray, net_error: np.ndarray,
                 snr: np.ndarray, aperture_area: np.ndarray,
                 coverage_fraction: np.ndarray, background_valid_pixels: np.ndarray):
        self.source_counts = source_counts
        self.source_error = source_error
        self.background = background
        self.background_error = background_error
        self.net_counts = net_counts
        self.net_error = net_error
        self.snr = snr
        self.aperture_area = aperture_area
        self.coverage_fraction = coverage_fraction
        self.background_valid_pixels = background_valid_pixels

    def to_dict(self) -> Dict[str, np.ndarray]:
        """Convert to dictionary."""
        return {
            'source_counts': self.source_counts,
            'source_error': self.source_error,
            'background': self.background,
            'background_error': self.background_error,
            'net_counts': self.net_counts,
            'net_error': self.net_error,
            'snr': self.snr,
            'aperture_area': self.aperture_area,
            'coverage_fraction': self.coverage_fraction,
            'background_valid_pixels': self.background_valid_pixels
        }


def _measure_source_photometry(
    data: np.ndarray,
    error_squared: np.ndarray,
    positions: np.ndarray,
    radii_pix: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Measure source counts, errors, and coverage fractions for per-source radius.
    """
    n_sources = len(positions)
    source_counts = np.full(n_sources, np.nan, dtype=float)
    source_error = np.full(n_sources, np.nan, dtype=float)
    coverage_fraction = np.full(n_sources, np.nan, dtype=float)

    valid = np.isfinite(radii_pix) & (radii_pix > 0)
    if not np.all(valid):
        invalid_total = np.sum(~valid)
        if invalid_total > 0:
            logger.warning(
                "Found %d sources with non-positive or invalid aperture radii; skipping photometry for them.",
                int(invalid_total)
            )

    for idx in np.where(valid)[0]:
        radius = float(radii_pix[idx])
        pos = np.atleast_2d(positions[idx])
        aperture = CircularAperture(pos, r=radius)
        mask_obj = aperture.to_mask(method='center')[0]

        data_cut = mask_obj.cutout(data)
        err_cut = mask_obj.cutout(error_squared)
        if data_cut is None or err_cut is None:
            continue

        weights = mask_obj.data
        pixel_mask = weights > 0
        if not np.any(pixel_mask):
            continue

        valid_mask = pixel_mask & np.isfinite(data_cut) & np.isfinite(err_cut)
        total_weight = np.sum(weights[pixel_mask])
        valid_weight = np.sum(weights[valid_mask])

        if total_weight <= 0 or valid_weight <= 0:
            continue

        coverage_fraction[idx] = valid_weight / total_weight

        weighted_data = data_cut[valid_mask] * weights[valid_mask]
        weighted_error = err_cut[valid_mask] * (weights[valid_mask] ** 2)

        source_counts[idx] = float(np.sum(weighted_data))
        source_error[idx] = float(np.sqrt(np.sum(weighted_error)))

    return source_counts, source_error, coverage_fraction


def perform_aperture_photometry(
    xray_map,
    ra: np.ndarray,
    dec: np.ndarray,
    aperture_radius: np.ndarray | float = 16.0,
    background_method: str = 'annulus',
    background_inner: np.ndarray | float = 20.0,
    background_outer: np.ndarray | float = 30.0,
    background_sigma_clip: Optional[float] = None,
    min_coverage_fraction: float = 0.5,
    verbose: bool = True
) -> PhotometryResult:
    """
    Perform aperture photometry on X-ray map at specified positions.

    Parameters
    ----------
    xray_map : XrayMap
        X-ray map object with data, error, and WCS
    ra : np.ndarray
        Right ascension of sources (degrees)
    dec : np.ndarray
        Declination of sources (degrees)
    aperture_radius : float or array-like, optional
        Aperture radius in arcseconds (scalar or per-source array)
    background_method : str, optional
        Background estimation method: 'annulus' or 'local_median' (default: 'annulus')
    background_inner : float or array-like, optional
        Inner radius of background annulus in arcseconds (scalar or per-source array)
    background_outer : float or array-like, optional
        Outer radius of background annulus in arcseconds (scalar or per-source array)
    background_sigma_clip : float, optional
        Sigma-clipping threshold applied within the background annulus to reject
        contaminating emission (only used when background_method='annulus').
    min_coverage_fraction : float, optional
        Minimum fraction of unmasked pixels required within the source aperture.
        Sources falling below this threshold are flagged and set to NaN (default: 0.5).
    verbose : bool, optional
        Print progress information

    Returns
    -------
    PhotometryResult
        Container with photometry results including counts, errors, and SNR
    """
    ra = np.asarray(ra, dtype=float)
    dec = np.asarray(dec, dtype=float)
    n_sources = len(ra)

    aperture_radius = np.asarray(aperture_radius, dtype=float)
    if aperture_radius.size == 1:
        aperture_radius = np.full(n_sources, aperture_radius.item(), dtype=float)
    elif aperture_radius.size != n_sources:
        raise ValueError("aperture_radius must be scalar or match number of sources")

    background_inner = np.asarray(background_inner, dtype=float)
    if background_inner.size == 1:
        background_inner = np.full(n_sources, background_inner.item(), dtype=float)
    elif background_inner.size != n_sources:
        raise ValueError("background_inner must be scalar or match number of sources")

    background_outer = np.asarray(background_outer, dtype=float)
    if background_outer.size == 1:
        background_outer = np.full(n_sources, background_outer.item(), dtype=float)
    elif background_outer.size != n_sources:
        raise ValueError("background_outer must be scalar or match number of sources")

    if verbose:
        logger.info(f"Performing aperture photometry for {n_sources} sources")
        logger.info(f"Aperture radius (median): {np.nanmedian(aperture_radius):.1f} arcsec")
        logger.info(f"Background method: {background_method}")

    # Convert aperture sizes from arcsec to pixels
    pixel_scale = xray_map.get_pixel_scale_arcsec()
    aperture_radius_pix = aperture_radius / pixel_scale
    background_inner_pix = background_inner / pixel_scale
    background_outer_pix = background_outer / pixel_scale

    if verbose:
        logger.info(f"Pixel scale: {pixel_scale:.3f} arcsec/pixel")
        logger.info(f"Aperture radius (median): {np.nanmedian(aperture_radius_pix):.2f} pixels")

    # Convert sky coordinates to pixel coordinates
    x_pix, y_pix = xray_map.world_to_pixel(ra, dec)

    # Create apertures
    positions = np.column_stack((x_pix, y_pix))

    # Build a contamination mask so background annuli ignore neighbouring groups
    # mask=True indicates pixels to exclude during background estimation
    contamination_mask = np.zeros_like(xray_map.data, dtype=bool)
    valid_for_mask = np.isfinite(aperture_radius_pix) & (aperture_radius_pix > 0)
    if np.any(valid_for_mask):
        for idx in np.where(valid_for_mask)[0]:
            aperture = CircularAperture([positions[idx]], r=float(aperture_radius_pix[idx]))
            aperture_mask = aperture.to_mask(method='center')[0]
            mask_image = aperture_mask.to_image(xray_map.data.shape)
            if mask_image is not None:
                contamination_mask |= mask_image > 0

    # Perform aperture photometry on data map
    error_squared = xray_map.error ** 2
    source_counts, source_error, coverage_fraction = _measure_source_photometry(
        xray_map.data,
        error_squared,
        positions,
        aperture_radius_pix
    )

    # Estimate background
    if background_method == 'annulus':
        background, background_error, background_valid_pixels = _estimate_background_annulus(
            xray_map,
            positions,
            background_inner_pix,
            background_outer_pix,
            error_squared,
            contamination_mask,
            sigma_clip=background_sigma_clip
        )
    elif background_method == 'local_median':
        background, background_error = _estimate_background_local_median(
            xray_map, positions, aperture_radius_pix, background_outer_pix
        )
        background_valid_pixels = np.full(len(ra), np.nan, dtype=float)
    else:
        raise ValueError(f"Unknown background method: {background_method}")

    # Calculate aperture area
    aperture_area = np.zeros(n_sources, dtype=float)
    valid_aperture_area = np.isfinite(aperture_radius_pix) & (aperture_radius_pix > 0)
    aperture_area[valid_aperture_area] = np.pi * aperture_radius_pix[valid_aperture_area]**2

    # Scale background to aperture area
    if background_method == 'annulus':
        if background_valid_pixels is not None and np.any(np.isfinite(background_valid_pixels)):
            pixel_area_arcsec = pixel_scale**2
            eff_annulus_area = background_valid_pixels * pixel_area_arcsec
        else:
            eff_annulus_area = np.pi * (background_outer_pix**2 - background_inner_pix**2)
        safe_annulus_area = np.where(eff_annulus_area > 0, eff_annulus_area, np.nan)
        scale_factor = np.where(np.isfinite(safe_annulus_area), aperture_area / safe_annulus_area, 0.0)
        background_scaled = background * scale_factor
        background_error_scaled = background_error * scale_factor
    else:
        background_scaled = background * aperture_area
        background_error_scaled = background_error * aperture_area

    min_coverage_fraction = float(min_coverage_fraction)
    invalid_coverage = (
        ~np.isfinite(coverage_fraction) |
        (coverage_fraction < min_coverage_fraction)
    )

    if np.any(invalid_coverage):
        logger.warning(
            "Aperture coverage below %.0f%% for %d sources; photometry set to NaN.",
            100 * min_coverage_fraction,
            int(np.sum(invalid_coverage))
        )
        source_counts[invalid_coverage] = np.nan
        source_error[invalid_coverage] = np.nan
        background_scaled[invalid_coverage] = np.nan
        background_error_scaled[invalid_coverage] = np.nan

    # Calculate net counts (background-subtracted)
    net_counts = source_counts - background_scaled

    # Propagate errors
    net_error = np.sqrt(source_error**2 + background_error_scaled**2)

    # Calculate signal-to-noise ratio
    snr = np.full(n_sources, np.nan, dtype=float)
    valid_snr_mask = np.isfinite(net_error) & (net_error > 0)
    snr[valid_snr_mask] = net_counts[valid_snr_mask] / net_error[valid_snr_mask]

    if verbose:
        n_detected = np.sum((snr > 3.0) & np.isfinite(snr))
        logger.info(f"Sources with SNR > 3: {n_detected}/{n_sources} "
                   f"({100*n_detected/n_sources:.1f}%)")
        logger.info(f"Median net counts: {np.nanmedian(net_counts):.2f}")
        logger.info(f"Median SNR: {np.nanmedian(snr):.2f}")

    return PhotometryResult(
        source_counts=source_counts,
        source_error=source_error,
        background=background_scaled,
        background_error=background_error_scaled,
        net_counts=net_counts,
        net_error=net_error,
        snr=snr,
        aperture_area=aperture_area,
        coverage_fraction=coverage_fraction,
        background_valid_pixels=background_valid_pixels
    )


def _estimate_background_annulus(
    xray_map,
    positions: np.ndarray,
    inner_radius,
    outer_radius,
    error_squared: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    sigma_clip: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate background using annulus around each source.

    Parameters
    ----------
    xray_map : XrayMap
        X-ray map object
    positions : np.ndarray
        Array of (x, y) pixel positions
    inner_radius : float
        Inner radius in pixels
    outer_radius : float
        Outer radius in pixels
    error_squared : np.ndarray, optional
        Precomputed squared error map to avoid recalculation
    mask : np.ndarray, optional
        Boolean mask marking pixels to exclude (True = ignore), e.g., other group apertures
    sigma_clip : float, optional
        Sigma-clipping threshold applied to annulus pixels to suppress contaminating
        emission (values > median + sigma_clip * std are excluded)

    Returns
    -------
    background : np.ndarray
        Background counts in annulus
    background_error : np.ndarray
        Error on background counts
    """
    inner_radius = np.asarray(inner_radius, dtype=float)
    outer_radius = np.asarray(outer_radius, dtype=float)
    n_sources = len(positions)

    background = np.zeros(n_sources, dtype=float)
    background_error = np.zeros(n_sources, dtype=float)
    valid_pixel_counts = np.full(n_sources, np.nan, dtype=float)

    if error_squared is None:
        error_squared = xray_map.error ** 2

    valid = (
        np.isfinite(inner_radius) &
        np.isfinite(outer_radius) &
        (inner_radius >= 0) &
        (outer_radius > inner_radius)
    )

    if not np.any(valid):
        if np.sum(~valid) == n_sources:
            logger.warning("All sources have invalid background annulus parameters; background set to zero.")
        return background, background_error, valid_pixel_counts

    invalid_count = np.sum(~valid)
    if invalid_count > 0:
        logger.warning(
            "Found %d sources with invalid background annulus radii; background set to zero for them.",
            int(invalid_count)
        )

    for idx in np.where(valid)[0]:
        rin = float(inner_radius[idx])
        rout = float(outer_radius[idx])
        pos = np.atleast_2d(positions[idx])
        annulus = CircularAnnulus(pos, r_in=rin, r_out=rout)

        mask_obj = annulus.to_mask(method='center')[0]

        data_cut = mask_obj.cutout(xray_map.data)
        err_cut = mask_obj.cutout(error_squared)

        if data_cut is None or err_cut is None:
            continue

        mask_data = mask_obj.data
        pixel_mask = mask_data > 0

        if mask is not None:
            mask_cut = mask_obj.cutout(mask.astype(float))
            if mask_cut is not None:
                pixel_mask &= mask_cut == 0

        if not np.any(pixel_mask):
            continue

        pixel_mask &= np.isfinite(data_cut) & np.isfinite(err_cut)

        if not np.any(pixel_mask):
            continue

        if sigma_clip is not None and sigma_clip > 0:
            # Adaptive iterative sigma clipping: try multiple thresholds to find one that
            # retains enough pixels (addresses issue where 1.5σ removes 99.6% of pixels
            # when contamination is non-uniform but visually 40-50% of annulus is usable)
            # Save original mask before sigma clipping
            original_pixel_mask = pixel_mask.copy()
            initial_pixel_count = np.count_nonzero(original_pixel_mask)
            expected_pixels = initial_pixel_count  # Approximate expected pixels in annulus
            
            # Try thresholds from least to most aggressive
            # User observation: 40-50% visually usable, but 1.5σ gives only 0.4%
            # Try 2.5σ first (should give ~10%), then 2.0σ (~5%), then 1.5σ (current)
            thresholds_to_try = [2.5, 2.0, float(sigma_clip)]
            thresholds_to_try = sorted(set(thresholds_to_try), reverse=True)  # Descending order
            
            min_fraction = 0.05  # Require at least 5% pixels survive
            best_pixel_mask = None
            best_threshold_used = None
            
            for threshold in thresholds_to_try:
                # Reset pixel mask to original for this threshold attempt
                test_pixel_mask = original_pixel_mask.copy()
                
                # Iterative sigma clipping with this threshold
                max_iterations = 10
                iteration = 0
                pixels_removed_this_iter = True
                
                while pixels_removed_this_iter and iteration < max_iterations:
                    coords = np.where(test_pixel_mask)
                    if len(coords[0]) == 0:
                        break
                        
                    values = data_cut[coords]
                    if values.size == 0:
                        break
                    
                    median = float(np.nanmedian(values))
                    std = float(np.nanstd(values))
                    
                    if not (np.isfinite(std) and std > 0):
                        break
                    
                    # Remove pixels outside threshold * std from median
                    keep = np.abs(values - median) <= threshold * std
                    
                    if np.any(~keep):
                        test_pixel_mask[coords[0][~keep], coords[1][~keep]] = False
                        pixels_removed_this_iter = True
                        iteration += 1
                    else:
                        pixels_removed_this_iter = False
                    
                    if not np.any(test_pixel_mask):
                        break
                
                # Check if this threshold retained enough pixels
                pixels_survived = np.count_nonzero(test_pixel_mask)
                fraction = pixels_survived / expected_pixels if expected_pixels > 0 else 0.0
                
                if fraction >= min_fraction:
                    # This threshold is good - use it
                    pixel_mask = test_pixel_mask
                    best_threshold_used = threshold
                    logger.debug(
                        "Adaptive sigma clipping for source %d: using %.1fσ threshold "
                        "(%.1f%% pixels survive, %d pixels)",
                        idx, threshold, fraction * 100, pixels_survived
                    )
                    break
                else:
                    # This threshold too aggressive, but keep as fallback
                    if best_pixel_mask is None or pixels_survived > np.count_nonzero(best_pixel_mask):
                        best_pixel_mask = test_pixel_mask
                        best_threshold_used = threshold
            
            # Use best result (either one that met min_fraction, or best fallback)
            if best_pixel_mask is not None and np.any(best_pixel_mask):
                pixel_mask = best_pixel_mask
                final_pixels = np.count_nonzero(pixel_mask)
                final_fraction = final_pixels / expected_pixels if expected_pixels > 0 else 0.0
                
                if final_fraction < min_fraction:
                    logger.debug(
                        "Adaptive sigma clipping for source %d: all thresholds too aggressive "
                        "(%.1f%% pixels survive with %.1fσ). Using most lenient result.",
                        idx, final_fraction * 100, best_threshold_used
                    )
            else:
                # No pixels survived any threshold - fall back to original mask (no clipping)
                # This ensures we still measure background even if clipping is too aggressive
                logger.debug(
                    "Adaptive sigma clipping for source %d: all thresholds removed all pixels. "
                    "Using original mask without clipping.",
                    idx
                )
                pixel_mask = original_pixel_mask.copy()
            
            # Final check: if still no pixels, skip this source
            if not np.any(pixel_mask):
                logger.warning(
                    "No valid pixels in background annulus for source %d after all processing",
                    idx
                )
                continue

        data_values = data_cut[pixel_mask]
        err_values = err_cut[pixel_mask]

        background[idx] = float(np.nansum(data_values))
        background_error[idx] = float(np.sqrt(np.nansum(err_values)))
        valid_pixel_counts[idx] = float(np.count_nonzero(pixel_mask))

    valid_pixel_counts[valid_pixel_counts <= 0] = np.nan

    return background, background_error, valid_pixel_counts


def _estimate_background_local_median(
    xray_map,
    positions: np.ndarray,
    aperture_radius,
    outer_radius
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate background using local median in annulus.

    This method is more robust to contaminating sources but requires
    more computation.

    Parameters
    ----------
    xray_map : XrayMap
        X-ray map object
    positions : np.ndarray
        Array of (x, y) pixel positions
    aperture_radius : float
        Source aperture radius in pixels
    outer_radius : float
        Outer radius for background region in pixels

    Returns
    -------
    background : np.ndarray
        Background surface brightness (counts/pixel)
    background_error : np.ndarray
        Error on background surface brightness
    """
    n_sources = len(positions)
    aperture_radius = np.asarray(aperture_radius, dtype=float)
    if aperture_radius.size == 1:
        aperture_radius = np.full(n_sources, aperture_radius.item(), dtype=float)
    elif aperture_radius.size != n_sources:
        raise ValueError("aperture_radius must be scalar or match number of sources")

    outer_radius = np.asarray(outer_radius, dtype=float)
    if outer_radius.size == 1:
        outer_radius = np.full(n_sources, outer_radius.item(), dtype=float)
    elif outer_radius.size != n_sources:
        raise ValueError("outer_radius must be scalar or match number of sources")

    background = np.zeros(n_sources)
    background_error = np.zeros(n_sources)

    for i, (x, y) in enumerate(positions):
        xi, yi = int(np.round(x)), int(np.round(y))
        ar = aperture_radius[i]
        orad = outer_radius[i]

        if not np.isfinite(ar) or not np.isfinite(orad) or orad <= ar:
            background[i] = 0.0
            background_error[i] = 0.0
            continue

        size = int(max(1, np.ceil(orad)))
        y_min, y_max = max(0, yi - size), min(xray_map.shape[0], yi + size + 1)
        x_min, x_max = max(0, xi - size), min(xray_map.shape[1], xi + size + 1)

        yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
        dist = np.sqrt((xx - x)**2 + (yy - y)**2)

        mask = (dist > ar) & (dist <= orad)

        if np.sum(mask) > 0:
            annulus_data = xray_map.data[y_min:y_max, x_min:x_max][mask]
            annulus_error = xray_map.error[y_min:y_max, x_min:x_max][mask]

            valid = ~np.isnan(annulus_data)
            if np.sum(valid) > 0:
                background[i] = np.median(annulus_data[valid])
                background_error[i] = np.std(annulus_error[valid]) / np.sqrt(np.sum(valid))
            else:
                background[i] = 0.0
                background_error[i] = 0.0
        else:
            background[i] = 0.0
            background_error[i] = 0.0

    return background, background_error


def calculate_count_rate(
    net_counts: np.ndarray,
    exposure_time: float,
    net_error: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Convert counts to count rate.

    Parameters
    ----------
    net_counts : np.ndarray
        Net source counts
    exposure_time : float
        Exposure time in seconds
    net_error : np.ndarray, optional
        Error on net counts

    Returns
    -------
    count_rate : np.ndarray
        Count rate in counts/second
    count_rate_error : np.ndarray, optional
        Error on count rate
    """
    count_rate = net_counts / exposure_time

    if net_error is not None:
        count_rate_error = net_error / exposure_time
        return count_rate, count_rate_error
    else:
        return count_rate, None
