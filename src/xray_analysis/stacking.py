"""
Stacking analysis module.

Performs stacking analysis for low signal-to-noise sources, typically
binned by redshift or other properties.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class StackingResult:
    """Container for stacking analysis results."""

    def __init__(self, bin_edges: np.ndarray, bin_centers: np.ndarray,
                 n_sources: np.ndarray, stacked_signal: np.ndarray,
                 stacked_error: np.ndarray, snr: np.ndarray,
                 median_properties: Dict, is_valid: np.ndarray,
                 background_median: np.ndarray):
        self.bin_edges = bin_edges
        self.bin_centers = bin_centers
        self.n_sources = n_sources
        self.stacked_signal = stacked_signal
        self.stacked_error = stacked_error
        self.snr = snr
        self.median_properties = median_properties
        self.is_valid = is_valid
        self.background_median = background_median

    def to_dict(self) -> Dict:
        """Convert to dictionary.

        Note: bin_edges are stored as separate columns (bin_edge_lower, bin_edge_upper)
        to ensure all columns have the same length for table creation.
        """
        result = {
            'bin_edge_lower': self.bin_edges[:-1],
            'bin_edge_upper': self.bin_edges[1:],
            'bin_centers': self.bin_centers,
            'n_sources': self.n_sources,
            'stacked_signal': self.stacked_signal,
            'stacked_error': self.stacked_error,
            'snr': self.snr,
            'is_valid': self.is_valid,
            'background_median': self.background_median
        }
        result.update(self.median_properties)
        return result


def perform_stacking_analysis(
    xray_map,
    ra: np.ndarray,
    dec: np.ndarray,
    bin_variable: np.ndarray,
    bin_edges: np.ndarray,
    aperture_radius: float | np.ndarray = 16.0,
    method: str = 'median',
    sigma_clip: Optional[float] = 3.0,
    bootstrap_iterations: int = 1000,
    properties_dict: Optional[Dict[str, np.ndarray]] = None,
    min_sources_per_bin: int = 0,
    background_inner_factor: float = 1.5,
    background_outer_factor: float = 3.0,
    background_method: str = 'local',
    verbose: bool = True
) -> StackingResult:
    """
    Perform stacking analysis on X-ray sources binned by a variable.

    Parameters
    ----------
    xray_map : XrayMap
        X-ray map object
    ra : np.ndarray
        Right ascension array
    dec : np.ndarray
        Declination array
    bin_variable : np.ndarray
        Variable to bin by (e.g., redshift)
    bin_edges : np.ndarray
        Bin edges for stacking
    aperture_radius : float or np.ndarray, optional
        Aperture radius in arcseconds. If array, per-source aperture sizes.
        Must match length of ra/dec arrays.
    method : str, optional
        Stacking method: 'median' or 'mean'
    sigma_clip : float, optional
        Sigma clipping threshold (None to disable)
    bootstrap_iterations : int, optional
        Number of bootstrap iterations for error estimation
    properties_dict : dict, optional
        Dictionary of additional properties to track median values
    min_sources_per_bin : int, optional
        Minimum number of sources required per bin to treat the stack as valid.
        Bins that do not meet the threshold are flagged as invalid and filled with NaN.
    background_inner_factor : float, optional
        Inner radius scaling factor for local background annulus (default: 1.5 × aperture radius).
    background_outer_factor : float, optional
        Outer radius scaling factor for local background annulus (default: 3.0 × aperture radius).
    background_method : str, optional
        'local': subtract each source's own annulus median. 'bin_median': subtract the median
        of annulus levels in that bin for all sources (reduces impact of contaminated annuli).
    verbose : bool, optional
        Print progress

    Returns
    -------
    StackingResult
        Container with stacking results
    """
    if verbose:
        logger.info(f"Performing stacking analysis for {len(ra)} sources")
        logger.info(f"Binning by variable with {len(bin_edges)-1} bins")
        logger.info(f"Method: {method}, Sigma clip: {sigma_clip}, Background: {background_method}")

    n_bins = len(bin_edges) - 1
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Initialize arrays
    n_sources_per_bin = np.zeros(n_bins, dtype=int)
    stacked_signal = np.full(n_bins, np.nan, dtype=float)
    stacked_error = np.full(n_bins, np.nan, dtype=float)
    snr = np.full(n_bins, np.nan, dtype=float)
    is_valid = np.zeros(n_bins, dtype=bool)
    background_median = np.full(n_bins, np.nan, dtype=float)

    # Track median properties
    median_properties = {}
    if properties_dict is not None:
        for key in properties_dict.keys():
            median_properties[f'median_{key}'] = np.full(n_bins, np.nan, dtype=float)

    # Convert aperture to pixels (handle both scalar and array)
    pixel_scale = xray_map.get_pixel_scale_arcsec()
    aperture_radius = np.asarray(aperture_radius, dtype=float)
    if aperture_radius.size == 1:
        aperture_radius = np.full(len(ra), aperture_radius.item())
    elif aperture_radius.size != len(ra):
        raise ValueError(f"aperture_radius must be scalar or array of length {len(ra)}")
    
    aperture_radius_pix = aperture_radius / pixel_scale

    # Convert coordinates to pixels
    x_pix, y_pix = xray_map.world_to_pixel(ra, dec)

    # Stack in each bin
    for i in range(n_bins):
        if verbose:
            logger.info(f"Processing bin {i+1}/{n_bins}: "
                       f"[{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f}]")

        # Select sources in this bin
        mask = (bin_variable >= bin_edges[i]) & (bin_variable < bin_edges[i+1])
        n_sources_per_bin[i] = np.sum(mask)

        if n_sources_per_bin[i] == 0:
            logger.warning(f"Bin {i+1} has no sources, skipping")
            continue

        if n_sources_per_bin[i] < max(min_sources_per_bin, 0):
            if verbose:
                logger.warning(f"Bin {i+1} has {n_sources_per_bin[i]} sources "
                               f"(minimum required: {min_sources_per_bin}); marking as invalid")
            continue

        # Get pixel positions and aperture radii for this bin
        x_bin = x_pix[mask]
        y_bin = y_pix[mask]
        aperture_radius_pix_bin = aperture_radius_pix[mask]

        # Extract cutouts and compute net signal per source
        net_counts_values = []
        bin_background_values = []
        use_bin_median = (background_method == 'bin_median')
        raw_sums = []  # used when use_bin_median
        aperture_areas = []  # used when use_bin_median

        for x, y, rad_pix in zip(x_bin, y_bin, aperture_radius_pix_bin):
            # Use size factor for cutout based on aperture size
            size_factor = 3.0  # Standard cutout size
            cutout, cutout_error = _extract_cutout(
                xray_map, x, y, rad_pix, size_factor=size_factor
            )
            if cutout is None:
                continue
            cutout_bg, bg_level = _subtract_local_background(
                cutout,
                radius_pix=rad_pix,
                inner_factor=background_inner_factor,
                outer_factor=background_outer_factor
            )
            if use_bin_median:
                # Store raw aperture sum and area; need valid bg_level for median
                if bg_level is not None:
                    bin_background_values.append(bg_level)
                    raw_sums.append(_aperture_sum(cutout, rad_pix))
                    aperture_areas.append(np.pi * rad_pix ** 2)
            else:
                if cutout_bg is None:
                    continue
                if bg_level is not None:
                    bin_background_values.append(bg_level)
                net_counts_values.append(_aperture_sum(cutout_bg, rad_pix))

        if use_bin_median and len(bin_background_values) > 0 and len(raw_sums) == len(bin_background_values):
            median_bg = float(np.median(bin_background_values))
            raw_sums = np.asarray(raw_sums)
            aperture_areas = np.asarray(aperture_areas)
            net_counts_values = list(raw_sums - median_bg * aperture_areas)

        if len(net_counts_values) == 0:
            logger.warning(f"Bin {i+1} has no valid cutouts")
            continue

        if verbose:
            logger.info(f"  Valid cutouts: {len(net_counts_values)}/{n_sources_per_bin[i]}")

        net_counts_values = np.array(net_counts_values)
        if sigma_clip is not None and len(net_counts_values) > 1:
            median_val = np.median(net_counts_values)
            std_val = np.std(net_counts_values, ddof=1)
            if std_val > 0:
                clip_mask = np.abs(net_counts_values - median_val) <= sigma_clip * std_val
                if np.sum(clip_mask) > 0:
                    net_counts_values = net_counts_values[clip_mask]
        if method == 'median':
            stacked_signal[i] = float(np.median(net_counts_values))
        elif method == 'mean':
            stacked_signal[i] = float(np.mean(net_counts_values))
        else:
            raise ValueError(f"Unknown method: {method}")

        if verbose:
            logger.info(f"  Aperture sum: {stacked_signal[i]:.3e}")

        # Bootstrap error estimation
        stacked_error[i] = _bootstrap_error(
            net_counts_values,
            method=method,
            n_iterations=bootstrap_iterations
        )

        # Calculate SNR
        if stacked_error[i] > 0:
            snr[i] = stacked_signal[i] / stacked_error[i]

        # Calculate median properties
        if properties_dict is not None:
            for key, values in properties_dict.items():
                median_properties[f'median_{key}'][i] = np.median(values[mask])

        if verbose:
            logger.info(f"  N sources: {n_sources_per_bin[i]}")
            logger.info(f"  Stacked signal: {stacked_signal[i]:.4e}")
            logger.info(f"  SNR: {snr[i]:.2f}")

        is_valid[i] = True
        if len(bin_background_values) > 0:
            background_median[i] = float(np.median(bin_background_values))

    return StackingResult(
        bin_edges=bin_edges,
        bin_centers=bin_centers,
        n_sources=n_sources_per_bin,
        stacked_signal=stacked_signal,
        stacked_error=stacked_error,
        snr=snr,
        median_properties=median_properties,
        is_valid=is_valid,
        background_median=background_median
    )


def _extract_cutout(
    xray_map,
    x: float,
    y: float,
    radius_pix: float,
    size_factor: float = None
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Extract square cutout around a position.

    Parameters
    ----------
    xray_map : XrayMap
        X-ray map object
    x, y : float
        Pixel coordinates
    radius_pix : float
        Aperture radius in pixels
    size_factor : float, optional
        Cutout size factor (cutout_size = radius * size_factor). Default: 3.0

    Returns
    -------
    cutout : np.ndarray or None
        Cutout from data map
    cutout_error : np.ndarray or None
        Cutout from error map
    """
    if size_factor is None:
        size_factor = 3.0
    
    # Cutout size
    size = int(np.ceil(radius_pix * size_factor))

    # Get cutout boundaries
    xi, yi = int(np.round(x)), int(np.round(y))
    y_min, y_max = yi - size, yi + size + 1
    x_min, x_max = xi - size, xi + size + 1

    # Check boundaries
    if (x_min < 0 or x_max > xray_map.shape[1] or
        y_min < 0 or y_max > xray_map.shape[0]):
        return None, None

    # Extract cutout
    cutout = xray_map.data[y_min:y_max, x_min:x_max].copy()
    cutout_error = xray_map.error[y_min:y_max, x_min:x_max].copy()

    # Check for NaN values
    if np.any(np.isnan(cutout)):
        return None, None

    return cutout, cutout_error


def _sigma_clipped_median(data: List[np.ndarray], sigma: float) -> np.ndarray:
    """
    Calculate sigma-clipped median along axis 0.

    Parameters
    ----------
    data : list of np.ndarray
        List of arrays to stack
    sigma : float
        Sigma clipping threshold

    Returns
    -------
    median : np.ndarray
        Sigma-clipped median
    """
    data_array = np.array(data)

    # Iterative sigma clipping
    for _ in range(3):  # Maximum 3 iterations
        median = np.median(data_array, axis=0)
        std = np.std(data_array, axis=0)

        # Create mask for outliers
        mask = np.abs(data_array - median[np.newaxis, :, :]) > sigma * std[np.newaxis, :, :]

        # Replace outliers with median (broadcast median to match mask shape)
        median_broadcasted = np.broadcast_to(median[np.newaxis, :, :], data_array.shape)
        data_array[mask] = median_broadcasted[mask]

    return np.median(data_array, axis=0)


def _sigma_clipped_mean(data: List[np.ndarray], sigma: float) -> np.ndarray:
    """
    Calculate sigma-clipped mean along axis 0.

    Parameters
    ----------
    data : list of np.ndarray
        List of arrays to stack
    sigma : float
        Sigma clipping threshold

    Returns
    -------
    mean : np.ndarray
        Sigma-clipped mean
    """
    data_array = np.array(data)

    # Iterative sigma clipping
    for _ in range(3):
        mean = np.mean(data_array, axis=0)
        std = np.std(data_array, axis=0)

        mask = np.abs(data_array - mean[np.newaxis, :, :]) > sigma * std[np.newaxis, :, :]
        # Replace outliers with mean (broadcast mean to match mask shape)
        mean_broadcasted = np.broadcast_to(mean[np.newaxis, :, :], data_array.shape)
        data_array[mask] = mean_broadcasted[mask]

    return np.mean(data_array, axis=0)


def _aperture_sum(image: np.ndarray, radius_pix: float) -> float:
    """
    Calculate sum within circular aperture at image center.

    Parameters
    ----------
    image : np.ndarray
        2D image array
    radius_pix : float
        Aperture radius in pixels

    Returns
    -------
    aperture_sum : float
        Sum within aperture
    """
    # Get center
    cy, cx = np.array(image.shape) // 2

    # Create coordinate grids
    yy, xx = np.ogrid[:image.shape[0], :image.shape[1]]
    dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)

    # Create aperture mask
    mask = dist <= radius_pix

    # Calculate sum
    aperture_sum = np.sum(image[mask])

    return aperture_sum


def _bootstrap_error(
    values: np.ndarray,
    method: str,
    n_iterations: int
) -> float:
    """
    Estimate error on stacked signal using bootstrap resampling.

    Parameters
    ----------
    values : np.ndarray
        Per-source net signal within the aperture.
    method : str
        'median' or 'mean'
    n_iterations : int
        Number of bootstrap iterations

    Returns
    -------
    error : float
        Bootstrap error estimate
    """
    n_values = len(values)
    if n_values == 0 or n_iterations <= 0:
        return np.nan

    rng = np.random.default_rng()
    bootstrap_values = np.zeros(n_iterations, dtype=float)

    for i in range(n_iterations):
        # Resample with replacement
        resampled = rng.choice(values, size=n_values, replace=True)
        if method == 'median':
            bootstrap_values[i] = float(np.median(resampled))
        else:  # mean
            bootstrap_values[i] = float(np.mean(resampled))

    # Error is standard deviation of bootstrap distribution
    error = np.std(bootstrap_values, ddof=1)

    return error


def _subtract_local_background(
    cutout: np.ndarray,
    radius_pix: float,
    inner_factor: float,
    outer_factor: float
) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """
    Subtract local background estimated from an annulus around the source.

    Parameters
    ----------
    cutout : np.ndarray
        2D image cutout centred on the source.
    radius_pix : float
        Aperture radius in pixels.
    inner_factor : float
        Multiplicative factor to set the inner background radius.
    outer_factor : float
        Multiplicative factor to set the outer background radius.

    Returns
    -------
    background_subtracted_cutout : np.ndarray or None
        Cutout with local background subtracted. None if background estimate fails.
    background_level : float or None
        Estimated background surface brightness (counts/pixel). None if estimate fails.
    """
    if outer_factor <= inner_factor or inner_factor <= 1.0:
        outer_factor = max(outer_factor, inner_factor + 0.5)
        inner_factor = max(inner_factor, 1.05)

    cy, cx = np.array(cutout.shape) // 2
    yy, xx = np.ogrid[:cutout.shape[0], :cutout.shape[1]]
    dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)

    inner_radius = radius_pix * inner_factor
    outer_radius = radius_pix * outer_factor

    annulus_mask = (dist >= inner_radius) & (dist <= outer_radius)
    annulus_values = cutout[annulus_mask]

    valid = np.isfinite(annulus_values)
    if np.sum(valid) < 10:
        return None, None

    background_level = float(np.median(annulus_values[valid]))
    cutout_subtracted = cutout - background_level

    return cutout_subtracted, background_level


def stack_by_redshift(
    xray_map,
    ra: np.ndarray,
    dec: np.ndarray,
    redshift: np.ndarray,
    redshift_bins: np.ndarray,
    aperture_radius: float = 16.0,
    min_sources_per_bin: int = 5,
    background_inner_factor: float = 1.5,
    background_outer_factor: float = 3.0,
    background_method: str = 'local',
    **kwargs
) -> StackingResult:
    """
    Convenience function for stacking by redshift.

    Parameters
    ----------
    xray_map : XrayMap
        X-ray map object
    ra, dec : np.ndarray
        Coordinates
    redshift : np.ndarray
        Redshift array
    redshift_bins : np.ndarray
        Redshift bin edges
    aperture_radius : float
        Aperture radius in arcseconds
    min_sources_per_bin : int
        Minimum sources per bin (bins with fewer sources are skipped)
    background_inner_factor : float
        Inner radius factor for background annulus
    background_outer_factor : float
        Outer radius factor for background annulus
    background_method : str
        'local' (per-source annulus) or 'bin_median' (median in bin)
    **kwargs
        Additional arguments passed to perform_stacking_analysis

    Returns
    -------
    StackingResult
        Stacking results
    """
    # Count sources per bin
    bin_counts, _ = np.histogram(redshift, bins=redshift_bins)

    # Filter bins with sufficient sources
    valid_bins = bin_counts >= min_sources_per_bin
    logger.info(f"Bins with >= {min_sources_per_bin} sources: "
               f"{np.sum(valid_bins)}/{len(valid_bins)}")

    # Perform stacking
    result = perform_stacking_analysis(
        xray_map=xray_map,
        ra=ra,
        dec=dec,
        bin_variable=redshift,
        bin_edges=redshift_bins,
        aperture_radius=aperture_radius,
        properties_dict={'redshift': redshift},
        min_sources_per_bin=min_sources_per_bin,
        background_inner_factor=background_inner_factor,
        background_outer_factor=background_outer_factor,
        background_method=background_method,
        **kwargs
    )

    return result
