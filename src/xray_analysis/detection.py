"""
Detection significance module.

Calculates statistical significance of X-ray detections with proper
treatment of Poisson statistics and multiple testing corrections.
"""

import numpy as np
from scipy import stats
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class DetectionResult:
    """Container for detection significance results."""

    def __init__(self, is_detected: np.ndarray, significance: np.ndarray,
                 p_value: np.ndarray, snr: np.ndarray):
        self.is_detected = is_detected
        self.significance = significance
        self.p_value = p_value
        self.snr = snr

    def to_dict(self) -> Dict[str, np.ndarray]:
        """Convert to dictionary."""
        return {
            'is_detected': self.is_detected,
            'significance_sigma': self.significance,
            'p_value': self.p_value,
            'snr': self.snr
        }


def calculate_detection_significance(
    net_counts: np.ndarray,
    net_error: np.ndarray,
    snr_threshold: float = 3.0,
    min_counts: float = 0.0,
    method: str = 'gaussian',
    verbose: bool = True
) -> DetectionResult:
    """
    Calculate detection significance for X-ray sources.

    Parameters
    ----------
    net_counts : np.ndarray
        Net source counts (background-subtracted)
    net_error : np.ndarray
        Error on net counts
    snr_threshold : float, optional
        Signal-to-noise threshold for detection (default: 3.0)
    min_counts : float, optional
        Minimum net signal required for detection, in the same units as ``net_counts``
        (default: 0.0). When working with count-rate maps, supply a value in counts/s.
    method : str, optional
        Statistical method: 'gaussian' or 'poisson' (default: 'gaussian')
    verbose : bool, optional
        Print detection statistics

    Returns
    -------
    DetectionResult
        Container with detection results
    """
    if verbose:
        logger.info(f"Calculating detection significance for {len(net_counts)} sources")
        logger.info(f"Method: {method}, SNR threshold: {snr_threshold:.1f}, "
                    f"minimum net signal: {min_counts:.3e}")

    # Calculate SNR
    snr = np.where(net_error > 0, net_counts / net_error, 0.0)

    if method == 'gaussian':
        # Gaussian approximation (valid for high counts)
        significance = snr
        p_value = 1.0 - stats.norm.cdf(snr)

    elif method == 'poisson':
        # Poisson statistics (more accurate for low counts)
        significance, p_value = _poisson_significance(net_counts, net_error)

    else:
        raise ValueError(f"Unknown method: {method}")

    # Detection criteria: SNR above threshold AND minimum counts
    min_counts = max(min_counts, 0.0)
    is_detected = (snr >= snr_threshold) & (net_counts >= min_counts)

    if verbose:
        n_detected = np.sum(is_detected)
        logger.info(f"Detected sources: {n_detected}/{len(net_counts)} "
                   f"({100*n_detected/len(net_counts):.1f}%)")
        if n_detected > 0:
            logger.info(f"Median significance (detected): {np.median(significance[is_detected]):.2f} σ")
            logger.info(f"Median net signal (detected): {np.median(net_counts[is_detected]):.2f}")

    return DetectionResult(
        is_detected=is_detected,
        significance=significance,
        p_value=p_value,
        snr=snr
    )


def _poisson_significance(net_counts: np.ndarray, net_error: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate significance using Poisson statistics.

    Uses the Poisson probability that the observed counts are consistent
    with background fluctuation.

    Parameters
    ----------
    net_counts : np.ndarray
        Net source counts
    net_error : np.ndarray
        Error on net counts (approximates sqrt of background)

    Returns
    -------
    significance : np.ndarray
        Significance in units of Gaussian sigma
    p_value : np.ndarray
        Probability value
    """
    # Estimate background from error
    # For Poisson: error ≈ sqrt(background)
    background = net_error ** 2

    # Total counts = net + background
    total_counts = net_counts + background

    # Calculate probability of observing >= total_counts if only background present
    # P(N >= N_obs | background) = 1 - Poisson_CDF(N_obs - 1, background)
    p_value = np.zeros_like(net_counts)

    for i in range(len(net_counts)):
        if total_counts[i] > 0 and background[i] > 0:
            # Poisson survival function
            p_value[i] = stats.poisson.sf(total_counts[i] - 1, background[i])
        else:
            p_value[i] = 1.0

    # Convert p-value to Gaussian significance
    # Avoid numerical issues with very small p-values
    p_value = np.clip(p_value, 1e-100, 1.0)
    significance = -stats.norm.ppf(p_value)

    return significance, p_value


def apply_false_discovery_rate(
    p_values: np.ndarray,
    fdr_threshold: float = 0.05
) -> np.ndarray:
    """
    Apply False Discovery Rate (FDR) correction for multiple testing.

    Uses the Benjamini-Hochberg procedure to control the FDR.

    Parameters
    ----------
    p_values : np.ndarray
        Array of p-values
    fdr_threshold : float, optional
        FDR threshold (default: 0.05 = 5%)

    Returns
    -------
    is_significant : np.ndarray
        Boolean array indicating significant detections after FDR correction
    """
    n = len(p_values)

    # Sort p-values
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]

    # Find largest k such that P(k) <= (k/n) * FDR
    k_max = 0
    for k in range(n):
        if sorted_p[k] <= ((k + 1) / n) * fdr_threshold:
            k_max = k + 1

    # Mark significant detections
    is_significant = np.zeros(n, dtype=bool)
    if k_max > 0:
        is_significant[sorted_indices[:k_max]] = True

    logger.info(f"FDR correction: {np.sum(is_significant)}/{n} significant after correction")

    return is_significant


def calculate_upper_limits(
    net_counts: np.ndarray,
    net_error: np.ndarray,
    confidence_level: float = 0.95,
    method: str = 'bayesian'
) -> np.ndarray:
    """
    Calculate upper limits for non-detections.

    Parameters
    ----------
    net_counts : np.ndarray
        Net source counts
    net_error : np.ndarray
        Error on net counts
    confidence_level : float, optional
        Confidence level for upper limit (default: 0.95 = 95%)
    method : str, optional
        Method for calculating limits: 'bayesian' or 'frequentist' (default: 'bayesian')

    Returns
    -------
    upper_limits : np.ndarray
        Upper limits on net counts at specified confidence level
    """
    if method == 'bayesian':
        # Bayesian upper limit using Poisson likelihood
        # For small counts, use proper Poisson posterior
        upper_limits = np.zeros_like(net_counts)

        for i in range(len(net_counts)):
            if net_counts[i] <= 0:
                # Use Poisson quantile for background
                background = net_error[i] ** 2
                upper_limits[i] = stats.poisson.ppf(confidence_level, background)
            else:
                # Use Gaussian approximation for higher counts
                z_score = stats.norm.ppf(confidence_level)
                upper_limits[i] = net_counts[i] + z_score * net_error[i]

    elif method == 'frequentist':
        # Classical frequentist upper limit
        z_score = stats.norm.ppf(confidence_level)
        upper_limits = net_counts + z_score * net_error

    else:
        raise ValueError(f"Unknown method: {method}")

    # Ensure non-negative
    upper_limits = np.maximum(upper_limits, 0.0)

    return upper_limits


def calculate_detection_efficiency(
    snr: np.ndarray,
    snr_threshold: float = 3.0,
    bins: int = 20
) -> Dict[str, np.ndarray]:
    """
    Calculate detection efficiency as a function of SNR.

    Useful for understanding completeness of the survey.

    Parameters
    ----------
    snr : np.ndarray
        Signal-to-noise ratios
    snr_threshold : float, optional
        Detection threshold
    bins : int, optional
        Number of SNR bins

    Returns
    -------
    dict
        Dictionary with 'snr_bins', 'efficiency', and 'counts' arrays
    """
    snr_range = [0, max(10, np.max(snr))]
    snr_bins = np.linspace(snr_range[0], snr_range[1], bins + 1)
    snr_centers = 0.5 * (snr_bins[:-1] + snr_bins[1:])

    efficiency = np.zeros(bins)
    counts_per_bin = np.zeros(bins)

    for i in range(bins):
        mask = (snr >= snr_bins[i]) & (snr < snr_bins[i + 1])
        counts_per_bin[i] = np.sum(mask)
        if counts_per_bin[i] > 0:
            efficiency[i] = np.sum((snr[mask] >= snr_threshold)) / counts_per_bin[i]

    return {
        'snr_bins': snr_centers,
        'efficiency': efficiency,
        'counts': counts_per_bin
    }
