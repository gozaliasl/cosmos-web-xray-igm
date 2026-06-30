#!/usr/bin/env python3
"""
Main X-ray analysis pipeline for galaxy groups.

This script performs the complete X-ray analysis including:
1. Load data (group catalog and X-ray maps)
2. Aperture photometry
3. Detection significance
4. X-ray properties (luminosity, flux)
5. Temperature estimation
6. Stacking analysis
7. Visualization

Usage:
    python main_analysis.py [--config config_refined_z.yaml]
"""

import sys
import copy
import logging
import argparse
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import yaml
import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.io import fits
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM, LambdaCDM

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from xray_analysis.data_loader import (
    load_group_catalog, load_xray_maps, validate_coverage
)
from xray_analysis.photometry import perform_aperture_photometry
from xray_analysis.detection import (
    calculate_detection_significance, calculate_upper_limits
)
from xray_analysis.xray_properties import (
    calculate_xray_flux, calculate_xray_luminosity
)
from xray_analysis.spectral_model import estimate_temperature_from_luminosity_redshift
from xray_analysis.stacking import stack_by_redshift
from xray_analysis.peak_finding import find_xray_centroid, find_xray_peak
from xray_analysis.contamination import check_projected_contamination
from xray_analysis.visualization import (
    plot_xray_map, plot_detection_map, plot_luminosity_redshift,
    plot_stacking_results, plot_diagnostic_panel, plot_upper_limit_diagnostics
)
from xray_analysis.mass_estimation import (
    estimate_mass_from_temperature,
    estimate_mass_from_luminosity
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def _compute_log10_with_error(values: np.ndarray, errors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    errors = np.asarray(errors, dtype=float)
    log_values = np.full_like(values, np.nan, dtype=float)
    log_errors = np.full_like(errors, np.nan, dtype=float)

    mask = np.isfinite(values) & (values > 0)
    if np.any(mask):
        log_values[mask] = np.log10(values[mask])
        valid_error_mask = mask & np.isfinite(errors) & (errors > 0)
        if np.any(valid_error_mask):
            log_errors[valid_error_mask] = errors[valid_error_mask] / (
                values[valid_error_mask] * np.log(10)
            )

    return log_values, log_errors


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def slugify(value: str) -> str:
    """Create filesystem-friendly slug from catalog name."""
    slug = re.sub(r'[^a-zA-Z0-9]+', '_', value.strip().lower()).strip('_')
    return slug or 'catalog'


def prepare_output_dirs(config: dict, slug: str) -> Dict[str, Path]:
    """Prepare results/figures/stacking directories for a catalog analysis."""
    results_dir = Path(config['output']['results_dir']) / slug
    figures_dir = Path(config['output']['figures_dir']) / slug
    stacking_dir = Path(config['output']['stacking_dir']) / slug

    for path in [results_dir, figures_dir, stacking_dir]:
        path.mkdir(parents=True, exist_ok=True)

    return {
        'results': results_dir,
        'figures': figures_dir,
        'stacking': stacking_dir,
    }


def get_catalog_entries(config: dict) -> List[Dict[str, str]]:
    """Return list of catalog entries with name, path, and optional redshift_threshold."""
    data_cfg = config.get('data', {})
    entries = []

    if 'catalogs' in data_cfg and data_cfg['catalogs']:
        for entry in data_cfg['catalogs']:
            path = entry.get('group_catalog')
            if not path:
                continue
            name = entry.get('name') or Path(path).stem
            catalog_entry = {'name': name, 'path': path}
            # Include catalog-specific redshift_threshold if present
            if 'redshift_threshold' in entry:
                catalog_entry['redshift_threshold'] = entry['redshift_threshold']
            entries.append(catalog_entry)
    else:
        path = data_cfg.get('group_catalog')
        if not path:
            raise ValueError("No group catalog specified in configuration.")
        name = data_cfg.get('catalog_name') or Path(path).stem
        catalog_entry = {'name': name, 'path': path}
        # Include catalog-specific redshift_threshold if present
        if 'redshift_threshold' in data_cfg:
            catalog_entry['redshift_threshold'] = data_cfg['redshift_threshold']
        entries.append(catalog_entry)

    return entries


def write_comparison_summary(metrics_list: List[Dict], config: dict) -> None:
    """Write catalog comparison summary as CSV."""
    if not metrics_list:
        return

    results_dir = Path(config['output']['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = results_dir / 'catalog_comparison.csv'

    import csv

    fieldnames = list(metrics_list[0].keys())
    with open(comparison_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics_list:
            writer.writerow(row)

    logger.info("Saved catalog comparison summary to %s", comparison_path)


def build_cosmology(config: dict) -> FlatLambdaCDM:
    cosmo_cfg = config.get('cosmology', {})
    H0 = cosmo_cfg.get('H0', 70.0)
    Om0 = cosmo_cfg.get('Om0', 0.3)
    Ode0 = cosmo_cfg.get('Ode0')
    if Ode0 is not None:
        total = Om0 + Ode0
        if abs(total - 1.0) < 1e-3:
            return FlatLambdaCDM(H0=H0, Om0=Om0)
        return LambdaCDM(H0=H0, Om0=Om0, Ode0=Ode0)
    return FlatLambdaCDM(H0=H0, Om0=Om0)


def compute_aperture_background_arrays(
    redshift: np.ndarray,
    config: dict,
    cosmology: FlatLambdaCDM
) -> Dict[str, np.ndarray]:
    phot_cfg = config.get('photometry', {})
    mode = phot_cfg.get('aperture_mode', 'fixed').lower()

    n = len(redshift)
    default_aperture = float(phot_cfg.get('aperture_radius_arcsec', 16.0))
    default_bg_inner = float(phot_cfg.get('background_inner_arcsec', 25.0))
    default_bg_outer = float(phot_cfg.get('background_outer_arcsec', 30.0))

    aperture_arcsec = np.full(n, default_aperture, dtype=float)
    aperture_kpc = np.full(n, np.nan, dtype=float)
    bg_inner_arcsec = np.full(n, default_bg_inner, dtype=float)
    bg_outer_arcsec = np.full(n, default_bg_outer, dtype=float)
    bg_inner_kpc = np.full(n, np.nan, dtype=float)
    bg_outer_kpc = np.full(n, np.nan, dtype=float)

    min_arcsec = float(phot_cfg.get('min_aperture_arcsec', 0.1))
    max_arcsec = float(phot_cfg.get('max_aperture_arcsec', 3600.0))
    gap_arcsec = float(phot_cfg.get('background_gap_arcsec', 0.0))

    if mode == 'physical_kpc':
        physical_radius_kpc = float(phot_cfg.get('physical_radius_kpc', 300.0))
        bg_inner_kpc_val = float(phot_cfg.get('background_inner_kpc', 500.0))
        bg_outer_kpc_val = float(phot_cfg.get('background_outer_kpc', 600.0))

        redshift_arr = np.asarray(redshift, dtype=float)
        valid = np.isfinite(redshift_arr) & (redshift_arr > 0)
        da = np.full(n, np.nan, dtype=float)
        if np.any(valid):
            da[valid] = cosmology.angular_diameter_distance(redshift_arr[valid]).to(u.kpc).value

        factor = u.rad.to(u.arcsec)
        with np.errstate(divide='ignore', invalid='ignore'):
            aperture_arcsec_valid = (physical_radius_kpc / da) * factor
            bg_inner_arcsec_valid = (bg_inner_kpc_val / da) * factor
            bg_outer_arcsec_valid = (bg_outer_kpc_val / da) * factor

        aperture_arcsec = np.where(np.isfinite(aperture_arcsec_valid), aperture_arcsec_valid, aperture_arcsec)
        bg_inner_arcsec = np.where(np.isfinite(bg_inner_arcsec_valid), bg_inner_arcsec_valid, bg_inner_arcsec)
        bg_outer_arcsec = np.where(np.isfinite(bg_outer_arcsec_valid), bg_outer_arcsec_valid, bg_outer_arcsec)

        aperture_kpc = np.where(np.isfinite(da), physical_radius_kpc, aperture_kpc)
        bg_inner_kpc = np.where(np.isfinite(da), bg_inner_kpc_val, bg_inner_kpc)
        bg_outer_kpc = np.where(np.isfinite(da), bg_outer_kpc_val, bg_outer_kpc)

    aperture_arcsec = np.clip(aperture_arcsec, min_arcsec, max_arcsec)
    bg_inner_arcsec = np.clip(bg_inner_arcsec, min_arcsec, np.inf)
    bg_inner_arcsec = np.maximum(bg_inner_arcsec, aperture_arcsec + gap_arcsec)
    bg_outer_arcsec = np.clip(bg_outer_arcsec, bg_inner_arcsec + gap_arcsec, np.inf)

    return {
        'aperture_arcsec': aperture_arcsec,
        'aperture_kpc': aperture_kpc,
        'background_inner_arcsec': bg_inner_arcsec,
        'background_outer_arcsec': bg_outer_arcsec,
        'background_inner_kpc': bg_inner_kpc,
        'background_outer_kpc': bg_outer_kpc,
        'aperture_mode': mode,
        'physical_radius_kpc': phot_cfg.get('physical_radius_kpc', np.nan),
        'min_arcsec': min_arcsec,
        'max_arcsec': max_arcsec,
        'gap_arcsec': gap_arcsec,
    }


def compute_extent_arcsec(
    catalog,
    coverage_mask: np.ndarray,
    redshift: np.ndarray,
    config: dict,
    cosmology: FlatLambdaCDM,
    xray_map
) -> Optional[np.ndarray]:
    phot_cfg = config.get('photometry', {})
    extent_cfg = phot_cfg.get('extent', {})
    column = extent_cfg.get('column')
    if not column:
        return None

    try:
        raw_values = np.asarray(catalog.data[column])
    except (AttributeError, KeyError):
        logger.info("Extent column '%s' not found in catalog; using fallback aperture strategy.", column)
        return None

    values = raw_values[coverage_mask]
    extent_arcsec = np.array(values, dtype=float)

    units = str(extent_cfg.get('units', 'arcsec')).lower()
    factor_rad_to_arcsec = u.rad.to(u.arcsec)

    if units in ('arcsec', 'arcseconds'):
        pass
    elif units in ('arcmin', 'arcminutes'):
        extent_arcsec = extent_arcsec * 60.0
    elif units in ('kpc', 'kiloparsec', 'kiloparsecs'):
        da = cosmology.angular_diameter_distance(redshift).to(u.kpc).value
        with np.errstate(divide='ignore', invalid='ignore'):
            extent_arcsec = (extent_arcsec / da) * factor_rad_to_arcsec
    elif units in ('pixel', 'pixels', 'pix'):
        pixel_scale = xray_map.get_pixel_scale_arcsec()
        extent_arcsec = extent_arcsec * pixel_scale
    else:
        logger.warning("Unknown extent units '%s'; skipping extent aperture mode", units)
        return None

    scale_factor = float(extent_cfg.get('scale_factor', 1.0))
    extent_arcsec *= scale_factor

    min_arcsec = extent_cfg.get('min_arcsec')
    max_arcsec = extent_cfg.get('max_arcsec')
    if min_arcsec is not None:
        extent_arcsec = np.where(np.isfinite(extent_arcsec), np.maximum(extent_arcsec, float(min_arcsec)), extent_arcsec)
    if max_arcsec is not None:
        extent_arcsec = np.where(np.isfinite(extent_arcsec), np.minimum(extent_arcsec, float(max_arcsec)), extent_arcsec)

    return extent_arcsec


def load_xray_maps_from_config(config: dict) -> Dict:
    """
    Load X-ray maps from configuration.
    Supports both single map (backward compatible) and dual map (redshift-based) modes.
    
    Returns:
        dict: Dictionary with 'full' and/or 'masked' keys containing XrayMap objects,
              or 'single' key if using single map mode.
    """
    data_cfg = config.get('data', {})
    verbose = config.get('analysis', {}).get('verbose', True)
    
    # Check for dual map configuration
    if 'xray_maps' in data_cfg:
        maps = {}
        if 'full' in data_cfg['xray_maps']:
            full_cfg = data_cfg['xray_maps']['full']
            maps['full'] = load_xray_maps(
                full_cfg['map'],
                full_cfg['error'],
                verbose=verbose
            )
            logger.info("Loaded full X-ray map (for z < threshold)")
        if 'masked' in data_cfg['xray_maps']:
            masked_cfg = data_cfg['xray_maps']['masked']
            maps['masked'] = load_xray_maps(
                masked_cfg['map'],
                masked_cfg['error'],
                verbose=verbose
            )
            logger.info("Loaded masked X-ray map (for z >= threshold)")
        return maps
    # Backward compatible: single map
    elif 'xray_map' in data_cfg and 'xray_error_map' in data_cfg:
        single_map = load_xray_maps(
            data_cfg['xray_map'],
            data_cfg['xray_error_map'],
            verbose=verbose
        )
        return {'single': single_map}
    else:
        raise ValueError("No X-ray map configuration found. Specify either 'xray_map'/'xray_error_map' or 'xray_maps' with 'full'/'masked' entries.")


def refine_detected_groups(
    analysis: Dict,
    ra: np.ndarray,
    dec: np.ndarray,
    redshift: np.ndarray,
    aperture_arcsec: np.ndarray,
    background_inner_arcsec: np.ndarray,
    background_outer_arcsec: np.ndarray,
    xray_maps: Dict,
    cosmology,
    config: dict,
    use_redshift_selection: bool,
    low_z_mask: np.ndarray = None,
    high_z_mask: np.ndarray = None,
    redshift_threshold: float = None,
    min_arcsec: float = None,
    max_arcsec: float = None,
    gap_arcsec: float = 5.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Refine detected groups with adaptive apertures (R500/R200) and X-ray peak centering.
    
    Returns updated ra, dec, aperture_arcsec, background_inner_arcsec, background_outer_arcsec, and updated analysis dict.
    """
    detected_mask = analysis['det_result'].is_detected
    n_detected = np.sum(detected_mask)
    
    if n_detected == 0:
        return ra, dec, aperture_arcsec, background_inner_arcsec, background_outer_arcsec, analysis
    
    logger.info("Refining %d detected groups with adaptive apertures and X-ray peak centering", n_detected)
    
    # Get R500/R200 from mass estimates
    r500_kpc = analysis['mass_from_temp'].R500.copy()
    r200_kpc = analysis['mass_from_temp'].R200.copy()
    # Fill in missing values with luminosity-based estimates
    missing_r500 = ~np.isfinite(r500_kpc) | (r500_kpc <= 0)
    missing_r200 = ~np.isfinite(r200_kpc) | (r200_kpc <= 0)
    if np.any(missing_r500):
        r500_kpc[missing_r500] = analysis['mass_from_lum'].R500[missing_r500]
    if np.any(missing_r200):
        r200_kpc[missing_r200] = analysis['mass_from_lum'].R200[missing_r200]
    
    # Convert R500/R200 to arcsec and update apertures
    updated_aperture_arcsec = aperture_arcsec.copy()
    updated_bg_inner_arcsec = background_inner_arcsec.copy()
    updated_bg_outer_arcsec = background_outer_arcsec.copy()
    
    factor = u.rad.to(u.arcsec)
    for idx in np.where(detected_mask)[0]:
        z_val = redshift[idx]
        if not (np.isfinite(z_val) and z_val > 0):
            continue
        
        # Choose target radius (R500 preferred, R200 fallback)
        if np.isfinite(r500_kpc[idx]) and r500_kpc[idx] > 0:
            target_radius_kpc = r500_kpc[idx]
        elif np.isfinite(r200_kpc[idx]) and r200_kpc[idx] > 0:
            target_radius_kpc = r200_kpc[idx]
        else:
            continue
        
        # Convert to arcsec
        da_kpc = cosmology.angular_diameter_distance(z_val).to(u.kpc).value
        target_radius_arcsec = (target_radius_kpc / da_kpc) * factor
        
        # Apply constraints
        if min_arcsec is not None:
            target_radius_arcsec = max(target_radius_arcsec, min_arcsec)
        if max_arcsec is not None:
            target_radius_arcsec = min(target_radius_arcsec, max_arcsec)
        
        # Update if significantly different (>10%)
        if abs(target_radius_arcsec - aperture_arcsec[idx]) / aperture_arcsec[idx] > 0.1:
            updated_aperture_arcsec[idx] = target_radius_arcsec
            updated_bg_inner_arcsec[idx] = max(background_inner_arcsec[idx], target_radius_arcsec + gap_arcsec)
            updated_bg_outer_arcsec[idx] = max(background_outer_arcsec[idx], updated_bg_inner_arcsec[idx] + gap_arcsec)
    
    # Find X-ray peaks for detected groups (only within R500 or R200 from analysis)
    updated_ra = ra.copy()
    updated_dec = dec.copy()
    # Arrays ra_xray_peak and dec_xray_peak are already initialized above
    n_peaks_found = 0
    pixel_scale = None

    for idx in np.where(detected_mask)[0]:
        # Only determine peak when we have R500 or R200 from X-ray analysis (search within that radius)
        if np.isfinite(r500_kpc[idx]) and r500_kpc[idx] > 0:
            peak_search_radius_kpc = r500_kpc[idx]
        elif np.isfinite(r200_kpc[idx]) and r200_kpc[idx] > 0:
            peak_search_radius_kpc = r200_kpc[idx]
        else:
            continue

        z_val = redshift[idx]
        if not (np.isfinite(z_val) and z_val > 0):
            continue
        da_kpc = cosmology.angular_diameter_distance(z_val).to(u.kpc).value
        if not (np.isfinite(da_kpc) and da_kpc > 0):
            continue
        peak_search_radius_arcsec = (peak_search_radius_kpc / da_kpc) * factor

        # Determine which map to use
        if use_redshift_selection:
            if low_z_mask is not None and low_z_mask[idx]:
                xray_map_for_peak = xray_maps['full']
            elif high_z_mask is not None and high_z_mask[idx]:
                xray_map_for_peak = xray_maps['masked']
            else:
                continue
        else:
            xray_map_for_peak = xray_maps.get('single') or xray_maps.get('full') or xray_maps.get('masked')

        if pixel_scale is None:
            pixel_scale = xray_map_for_peak.get_pixel_scale_arcsec()
        aperture_radius_pix = peak_search_radius_arcsec / pixel_scale

        # Convert to pixel coordinates
        x_pix, y_pix = xray_map_for_peak.world_to_pixel(np.array([ra[idx]]), np.array([dec[idx]]))
        initial_x = float(x_pix[0])
        initial_y = float(y_pix[0])
        
        # Find X-ray centroid (flux-weighted) within R500/R200; fallback to local peak finder if centroid fails
        peak_x, peak_y, peak_snr = find_xray_centroid(
            data=xray_map_for_peak.data,
            initial_x=initial_x,
            initial_y=initial_y,
            aperture_radius_pix=aperture_radius_pix,
            smoothing_sigma=2.0,
            min_snr=2.0,
            error_map=xray_map_for_peak.error if hasattr(xray_map_for_peak, 'error') else None
        )
        if peak_x is None or peak_y is None:
            peak_x, peak_y, peak_snr = find_xray_peak(
                data=xray_map_for_peak.data,
                initial_x=initial_x,
                initial_y=initial_y,
                search_radius_pix=min(50.0, aperture_radius_pix),
                smoothing_sigma=2.0,
                min_snr=1.5,
                error_map=xray_map_for_peak.error if hasattr(xray_map_for_peak, 'error') else None
            )
        
        if peak_x is not None and peak_y is not None:
            # Convert back to world coordinates
            if hasattr(xray_map_for_peak, 'wcs') and xray_map_for_peak.wcs is not None:
                try:
                    peak_coords = xray_map_for_peak.wcs.pixel_to_world(peak_x, peak_y)
                    peak_ra_deg = float(peak_coords.ra.deg)
                    peak_dec_deg = float(peak_coords.dec.deg)
                    # Store peak coordinates separately
                    ra_xray_peak[idx] = peak_ra_deg
                    dec_xray_peak[idx] = peak_dec_deg
                    # Update positions for refined photometry
                    updated_ra[idx] = peak_ra_deg
                    updated_dec[idx] = peak_dec_deg
                    n_peaks_found += 1
                except Exception as e:
                    logger.debug(f"Failed to convert peak to world coords for group {idx}: {e}")
    
    logger.info("Found X-ray peaks for %d/%d detected groups (within R500/R200)", n_peaks_found, n_detected)
    
    # Re-run photometry for detected groups if anything changed
    aperture_changed = not np.allclose(updated_aperture_arcsec, aperture_arcsec)
    position_changed = not (np.allclose(updated_ra, ra) and np.allclose(updated_dec, dec))
    
    if aperture_changed or position_changed:
        logger.info("Re-running photometry for detected groups with refined parameters")
        
        # Get the run_photometry_sequence function (defined in outer scope)
        # We'll need to call it with the right parameters
        
        # For now, return updated values - the caller will handle re-running photometry
        # This is a simplified version - full implementation would re-run photometry here
        return updated_ra, updated_dec, updated_aperture_arcsec, updated_bg_inner_arcsec, updated_bg_outer_arcsec, analysis
    
    return ra, dec, aperture_arcsec, background_inner_arcsec, background_outer_arcsec, analysis


def compute_redshift_binned_median_background(
    phot_result,
    redshift: np.ndarray,
    config: dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute background as median of annulus background surface brightness using sliding redshift window.
    
    For each group at redshift z, finds groups within z ± delta_z and computes median background
    from those groups. This provides a more adaptive and redshift-specific background estimate
    compared to fixed wide bins.
    
    Returns net_counts_binned, net_error_binned, background_binned, snr_binned, redshift_bin_index.
    """
    n = len(redshift)
    source_counts = np.asarray(phot_result.source_counts, dtype=float)
    source_error = np.asarray(phot_result.source_error, dtype=float)
    background = np.asarray(phot_result.background, dtype=float)
    aperture_area = np.asarray(phot_result.aperture_area, dtype=float)
    background_error = np.asarray(phot_result.background_error, dtype=float)

    # Surface brightness (counts per pixel in aperture) = background / aperture_area
    with np.errstate(divide='ignore', invalid='ignore'):
        bg_surf = np.where(aperture_area > 0, background / aperture_area, np.nan)
        bg_surf_err = np.where(aperture_area > 0, background_error / aperture_area, np.nan)

    # Get configuration
    phot_cfg = config.get('photometry', {})
    delta_z = float(phot_cfg.get('background_redshift_delta_z', 0.05))  # Default: 0.05
    min_groups_per_window = int(phot_cfg.get('background_redshift_min_groups', 5))  # Minimum groups for robust median
    use_sliding_window = phot_cfg.get('background_redshift_sliding_window', True)  # Enable sliding window
    
    # Check if we should use old fixed-bin method (for backward compatibility)
    n_bins_cfg = phot_cfg.get('background_redshift_n_bins')
    use_fixed_bins = (n_bins_cfg is not None) or (not use_sliding_window)
    
    if use_fixed_bins:
        # Old method: fixed redshift bins
        if n_bins_cfg is not None:
            z_min, z_max = np.nanmin(redshift), np.nanmax(redshift)
            if not np.isfinite(z_min) or not np.isfinite(z_max) or z_max <= z_min:
                z_min, z_max = 0.0, 3.0
            n_bins = int(n_bins_cfg)
            bins = np.linspace(z_min, z_max, n_bins + 1)
        else:
            # Use stacking redshift_bins as edges
            bins = np.asarray(config.get('stacking', {}).get('redshift_bins', [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.8]), dtype=float)
            if bins.size < 2:
                bins = np.linspace(np.nanmin(redshift), np.nanmax(redshift), 6)

        # Bin index per source (0-based)
        bin_idx = np.digitize(redshift, bins) - 1
        bin_idx = np.clip(bin_idx, 0, len(bins) - 2)

        n_bins_actual = len(bins) - 1
        median_surf = np.full(n_bins_actual, np.nan, dtype=float)
        median_surf_err = np.full(n_bins_actual, np.nan, dtype=float)

        for b in range(n_bins_actual):
            mask = bin_idx == b
            if not np.any(mask):
                continue
            vals = bg_surf[mask]
            vals = vals[np.isfinite(vals)]
            if vals.size > 0:
                median_surf[b] = float(np.nanmedian(vals))
                # Error: MAD scaled to Gaussian sigma, or std/sqrt(n)
                mad = np.nanmedian(np.abs(vals - np.nanmedian(vals)))
                median_surf_err[b] = 1.4826 * mad / np.sqrt(vals.size) if vals.size > 0 else np.nan
            errs = bg_surf_err[mask]
            errs = errs[np.isfinite(errs)]
            if errs.size > 0 and not np.isfinite(median_surf_err[b]):
                median_surf_err[b] = float(np.nanmedian(errs))

        # Per-source background and net counts from binned median
        background_binned = np.full(n, np.nan, dtype=float)
        for i in range(n):
            b = bin_idx[i]
            if np.isfinite(median_surf[b]) and np.isfinite(aperture_area[i]) and aperture_area[i] > 0:
                background_binned[i] = median_surf[b] * aperture_area[i]

        err_bg_binned = np.full(n, np.nan, dtype=float)
        for i in range(n):
            b = bin_idx[i]
            if np.isfinite(median_surf_err[b]) and np.isfinite(aperture_area[i]) and aperture_area[i] > 0:
                err_bg_binned[i] = median_surf_err[b] * aperture_area[i]
    else:
        # New method: sliding redshift window (z ± delta_z)
        background_binned = np.full(n, np.nan, dtype=float)
        err_bg_binned = np.full(n, np.nan, dtype=float)
        bin_idx = np.full(n, -1, dtype=int)  # Store window index for reference
        
        # For each group, find groups within z ± delta_z
        for i in range(n):
            z_i = redshift[i]
            if not np.isfinite(z_i) or z_i < 0:
                continue
            
            # Find groups within z_i ± delta_z
            z_window_min = z_i - delta_z
            z_window_max = z_i + delta_z
            mask = (redshift >= z_window_min) & (redshift < z_window_max) & np.isfinite(bg_surf)
            
            # If too few groups, expand window (up to 2×delta_z)
            if np.sum(mask) < min_groups_per_window:
                expanded_delta_z = delta_z * 2.0
                z_window_min = z_i - expanded_delta_z
                z_window_max = z_i + expanded_delta_z
                mask = (redshift >= z_window_min) & (redshift < z_window_max) & np.isfinite(bg_surf)
            
            # Get background surface brightness values in this window
            vals = bg_surf[mask]
            vals = vals[np.isfinite(vals)]
            
            if vals.size >= min_groups_per_window:
                # Compute median surface brightness for this window
                median_surf_i = float(np.nanmedian(vals))
                
                # Error: MAD scaled to Gaussian sigma
                mad = np.nanmedian(np.abs(vals - np.nanmedian(vals)))
                median_surf_err_i = 1.4826 * mad / np.sqrt(vals.size) if vals.size > 0 else np.nan
                
                # If MAD-based error fails, use median of individual errors
                if not np.isfinite(median_surf_err_i):
                    errs = bg_surf_err[mask]
                    errs = errs[np.isfinite(errs)]
                    if errs.size > 0:
                        median_surf_err_i = float(np.nanmedian(errs))
                
                # Convert to background counts for this group
                if np.isfinite(median_surf_i) and np.isfinite(aperture_area[i]) and aperture_area[i] > 0:
                    background_binned[i] = median_surf_i * aperture_area[i]
                
                if np.isfinite(median_surf_err_i) and np.isfinite(aperture_area[i]) and aperture_area[i] > 0:
                    err_bg_binned[i] = median_surf_err_i * aperture_area[i]
                
                # Store a pseudo-bin index (based on which delta_z window was used)
                bin_idx[i] = 0 if np.sum(mask) >= min_groups_per_window else 1

    # Calculate net counts and SNR
    net_counts_binned = source_counts - background_binned
    net_error_binned = np.sqrt(source_error ** 2 + np.where(np.isfinite(err_bg_binned), err_bg_binned ** 2, 0))

    snr_binned = np.full(n, np.nan, dtype=float)
    valid = np.isfinite(net_error_binned) & (net_error_binned > 0)
    snr_binned[valid] = net_counts_binned[valid] / net_error_binned[valid]

    return net_counts_binned, net_error_binned, background_binned, snr_binned, bin_idx


def run_catalog_analysis(catalog_name: str, catalog_path: Path, config: dict, xray_maps: Dict, redshift_threshold: float = None) -> Dict:
    """Execute full analysis for a single catalog and return summary metrics.
    
    Parameters
    ----------
    catalog_name : str
        Name of the catalog
    catalog_path : Path
        Path to the catalog file
    config : dict
        Configuration dictionary
    xray_maps : Dict
        Dictionary containing X-ray maps
    redshift_threshold : float, optional
        Catalog-specific redshift threshold. If None, uses global default from config.
    """
    verbose = config.get('analysis', {}).get('verbose', True)
    catalog_path = Path(catalog_path)
    if not catalog_path.exists():
        raise FileNotFoundError(f"Catalog file not found: {catalog_path}")

    slug = slugify(catalog_name)
    dirs = prepare_output_dirs(config, slug)
    results_dir = dirs['results']
    figures_dir = dirs['figures']
    stacking_dir = dirs['stacking']

    logger.info("\n" + "=" * 70)
    logger.info("PROCESSING CATALOG: %s", catalog_name)
    logger.info("=" * 70)

    catalog = load_group_catalog(str(catalog_path), verbose=verbose)

    ra, dec = catalog.get_coordinates()
    redshift = catalog.get_redshifts()
    n_groups = catalog.n_groups
    
    # Try to extract Group ID from catalog if available
    group_id_full = None
    catalog_data = catalog.data
    for col_name in ["Group_ID", "ID", "group_id", "id", "GroupID", "GROUP_ID"]:
        if col_name in catalog_data.colnames:
            group_id_full = np.asarray(catalog_data[col_name])
            logger.info("Found Group ID column: %s", col_name)
            break

    logger.info("Loaded %d galaxy groups from %s", n_groups, catalog_path)

    # Determine map selection mode
    use_redshift_selection = 'full' in xray_maps and 'masked' in xray_maps
    # Use catalog-specific threshold if provided, otherwise use global default
    if redshift_threshold is None:
        redshift_threshold = config.get('data', {}).get('redshift_threshold', 1.5)
    logger.info("Using redshift threshold: z = %.2f (catalog-specific: %s)", 
                redshift_threshold, "yes" if redshift_threshold != config.get('data', {}).get('redshift_threshold', 1.5) else "no (using global default)")
    
    if use_redshift_selection:
        logger.info("Using redshift-based map selection (threshold z=%.2f)", redshift_threshold)
        # Check coverage for both maps
        coverage_full = np.array(validate_coverage(catalog, xray_maps['full']), dtype=bool)
        coverage_masked = np.array(validate_coverage(catalog, xray_maps['masked']), dtype=bool)
        # Groups must have coverage in at least one map
        coverage = coverage_full | coverage_masked
        # Determine which map to use for each group based on redshift
        z_array = np.array(redshift)
        use_full_map = (z_array < redshift_threshold) & coverage_full
        use_masked_map = (z_array >= redshift_threshold) & coverage_masked
        # Fallback: if preferred map doesn't have coverage, use the other
        use_full_map |= (~use_full_map & ~use_masked_map & coverage_full)
        use_masked_map |= (~use_full_map & ~use_masked_map & coverage_masked)
    else:
        # Single map mode (backward compatible)
        xray_map = xray_maps.get('single') or xray_maps.get('full') or xray_maps.get('masked')
        if xray_map is None:
            raise ValueError("No valid X-ray map found in configuration")
        coverage = np.array(validate_coverage(catalog, xray_map), dtype=bool)
        use_full_map = np.zeros(n_groups, dtype=bool)
        use_masked_map = np.zeros(n_groups, dtype=bool)

    n_with_coverage = int(np.sum(coverage))

    ra = np.array(ra)[coverage]
    dec = np.array(dec)[coverage]
    redshift = np.array(redshift)[coverage]
    # Original catalog/group center from input catalog (never overwritten).
    # Used for results['RA']/['DEC'] so plots and overlays center on the group center, not the X-ray peak.
    ra_catalog = np.array(ra, dtype=float, copy=True)
    dec_catalog = np.array(dec, dtype=float, copy=True)
    # Filter Group ID by coverage mask if available
    group_id = None
    if group_id_full is not None:
        group_id = np.array(group_id_full)[coverage]
    n_analyzed = len(ra)
    
    # Initialize arrays to store X-ray peak coordinates separately
    # These will be populated during peak-finding and added to the results table
    ra_xray_peak = np.full(n_analyzed, np.nan, dtype=float)
    dec_xray_peak = np.full(n_analyzed, np.nan, dtype=float)

    if use_redshift_selection:
        use_full_map = use_full_map[coverage]
        use_masked_map = use_masked_map[coverage]
        n_low_z = int(np.sum(use_full_map))
        n_high_z = int(np.sum(use_masked_map))
        logger.info("Groups with X-ray coverage: %d/%d", n_with_coverage, n_groups)
        logger.info("  Low-z groups (z < %.2f): %d (using full map)", redshift_threshold, n_low_z)
        logger.info("  High-z groups (z >= %.2f): %d (using masked map)", redshift_threshold, n_high_z)
        logger.info("Analyzing %d groups with X-ray coverage", n_analyzed)
    else:
        logger.info("Groups with X-ray coverage: %d/%d", n_with_coverage, n_groups)
        logger.info("Analyzing %d groups with X-ray coverage", n_analyzed)

    if n_analyzed == 0:
        logger.warning("No groups with X-ray coverage for catalog %s", catalog_name)
        return {
            'Catalog': catalog_name,
            'Catalog_Path': str(catalog_path),
            'Slug': slug,
            'Total_Groups': n_groups,
            'Groups_With_Coverage': n_with_coverage,
            'Analyzed_Groups': n_analyzed,
            'Detections': 0,
            'Detection_Rate_percent': 0.0,
            'Median_Flux_erg_cm2_s': float('nan'),
            'Median_Luminosity_erg_s': float('nan'),
            'Median_Temperature_keV': float('nan'),
            'Results_Dir': str(results_dir),
            'Figures_Dir': str(figures_dir),
            'Stacking_Dir': str(stacking_dir),
        }
    cosmology = build_cosmology(config)
    phot_cfg = config.get('photometry', {})
    aperture_mode = str(phot_cfg.get('aperture_mode', 'fixed')).lower()
    extent_cfg = phot_cfg.get('extent', {})
    extent_apply_raw = extent_cfg.get('apply_mode')
    if extent_apply_raw is None:
        extent_apply_mode = 'detected' if aperture_mode == 'extent' else 'none'
    else:
        extent_apply_mode = str(extent_apply_raw).lower()
    # Determine which map to use for extent computation (use full map as default)
    xray_map_for_extent = xray_maps.get('full') or xray_maps.get('single') or xray_maps.get('masked')
    if xray_map_for_extent is None:
        raise ValueError("No valid X-ray map found for extent computation")
    
    extent_arcsec_array = compute_extent_arcsec(
        catalog,
        coverage,
        redshift,
        config,
        cosmology,
        xray_map_for_extent
    )
    extent_available = extent_arcsec_array is not None and np.any(np.isfinite(extent_arcsec_array))
    if extent_arcsec_array is None:
        extent_arcsec_array = np.full(n_analyzed, np.nan, dtype=float)

    aperture_config_for_mode = config
    if aperture_mode == 'extent' and not extent_available:
        photometry_cfg = config.get('photometry', {})
        fallback_cfg = copy.deepcopy(config)
        fallback_phot_cfg = fallback_cfg.setdefault('photometry', {})
        fallback_phot_cfg['aperture_mode'] = 'physical_kpc'
        aperture_config_for_mode = fallback_cfg

        phys_radius = fallback_phot_cfg.get('physical_radius_kpc', photometry_cfg.get('physical_radius_kpc', 300.0))
        logger.warning(
            "Extent column unavailable for catalog %s; falling back to physical aperture of %.0f kpc for this run.",
            catalog_name,
            phys_radius,
        )
        if extent_apply_mode != 'none':
            extent_apply_mode = 'none'
            logger.info(
                "Extent application disabled for catalog %s until extent measurements are provided.",
                catalog_name,
            )
    elif aperture_mode == 'extent' and extent_available:
        logger.info(
            "Aperture mode 'extent' enabled; applying catalog extents with mode '%s'.",
            extent_apply_mode,
        )

    aperture_info = compute_aperture_background_arrays(redshift, aperture_config_for_mode, cosmology)
    aperture_arcsec = aperture_info['aperture_arcsec']
    background_inner_arcsec = aperture_info['background_inner_arcsec']
    background_outer_arcsec = aperture_info['background_outer_arcsec']
    min_arcsec = aperture_info.get('min_arcsec')
    max_arcsec = aperture_info.get('max_arcsec')
    gap_arcsec = aperture_info.get('gap_arcsec', 0.0)

    extent_valid_mask = np.isfinite(extent_arcsec_array) & (extent_arcsec_array > 0)

    if extent_apply_mode in ('all', 'always') and np.any(extent_valid_mask):
        aperture_arcsec = np.where(extent_valid_mask, extent_arcsec_array, aperture_arcsec)

    if min_arcsec is not None:
        aperture_arcsec = np.maximum(aperture_arcsec, float(min_arcsec))
    if max_arcsec is not None:
        aperture_arcsec = np.minimum(aperture_arcsec, float(max_arcsec))

    background_inner_arcsec = np.maximum(background_inner_arcsec, aperture_arcsec + gap_arcsec)
    background_outer_arcsec = np.maximum(background_outer_arcsec, background_inner_arcsec + gap_arcsec)

    detection_settings = config.get('detection', {})
    min_signal_threshold = detection_settings.get('min_count_rate')
    if min_signal_threshold is None:
        min_signal_threshold = detection_settings.get('min_counts', 0.0)
        if 'min_counts' in detection_settings:
            logger.warning(
                "Detection setting 'min_counts' is deprecated; use 'min_count_rate'."
            )

    def run_photometry_sequence(xray_map_to_use, ra_subset, dec_subset, redshift_subset, ap_arcsec, bg_inner_arcsec, bg_outer_arcsec):
        phot = perform_aperture_photometry(
            xray_map=xray_map_to_use,
            ra=ra_subset,
            dec=dec_subset,
            aperture_radius=ap_arcsec,
            background_method=phot_cfg['background_method'],
            background_inner=bg_inner_arcsec,
            background_outer=bg_outer_arcsec,
            background_sigma_clip=phot_cfg.get('background_sigma_clip'),
            min_coverage_fraction=float(phot_cfg.get('min_coverage_fraction', 0.5)),
            verbose=verbose
        )

        det_res = calculate_detection_significance(
            net_counts=phot.net_counts,
            net_error=phot.net_error,
            snr_threshold=detection_settings['snr_threshold'],
            min_counts=min_signal_threshold,
            verbose=verbose
        )

        upper_limits_local = calculate_upper_limits(
            phot.net_counts,
            phot.net_error,
            confidence_level=0.95
        )

        flux_local, flux_error_local = calculate_xray_flux(
            net_counts=phot.net_counts,
            net_error=phot.net_error,
            count_rate_to_flux=config['xray']['count_rate_to_flux'],
            verbose=verbose
        )

        xray_props_local = calculate_xray_luminosity(
            flux=flux_local,
            flux_error=flux_error_local,
            redshift=redshift_subset,
            energy_band_kev=config['xray']['energy_band_kev'],
            cosmology=cosmology,
            k_correction=True,
            verbose=verbose
        )

        flux_upper_limits_local, flux_upper_error_local = calculate_xray_flux(
            net_counts=upper_limits_local,
            net_error=phot.net_error,
            count_rate_to_flux=config['xray']['count_rate_to_flux'],
            verbose=False
        )

        xray_props_upper_local = calculate_xray_luminosity(
            flux=flux_upper_limits_local,
            flux_error=flux_upper_error_local,
            redshift=redshift_subset,
            energy_band_kev=config['xray']['energy_band_kev'],
            cosmology=cosmology,
            k_correction=True,
            verbose=False
        )

        temperature_local, temperature_error_local = estimate_temperature_from_luminosity_redshift(
            luminosity=xray_props_local.luminosity,
            redshift=redshift_subset,
            scaling_relation='kettula2015',  # Using Kettula et al. (2015) bias-corrected relation
            luminosity_error=xray_props_local.luminosity_error  # Propagate measurement error, not scatter
        )

        temp_for_mass_local = np.array(temperature_local, dtype=float)
        temp_for_mass_local[~np.isfinite(temp_for_mass_local) | (temp_for_mass_local <= 0)] = np.nan
        temp_err_for_mass_local = np.array(temperature_error_local, dtype=float)
        temp_err_for_mass_local[~np.isfinite(temp_err_for_mass_local) | (temp_err_for_mass_local < 0)] = np.nan

        mass_cfg_local = config.get('mass', {})
        temp_scaling = mass_cfg_local.get('temperature_scaling', 'sun2009')
        lum_scaling = mass_cfg_local.get('luminosity_scaling', 'leauthaud2010')

        mass_from_temp_local = estimate_mass_from_temperature(
            temperature=temp_for_mass_local,
            temperature_error=temp_err_for_mass_local,
            redshift=redshift_subset,
            scaling_relation=temp_scaling,
            cosmology=cosmology,
            verbose=verbose
        )

        lum_for_mass_local = np.array(xray_props_local.luminosity, dtype=float)
        lum_for_mass_local[~np.isfinite(lum_for_mass_local) | (lum_for_mass_local <= 0)] = np.nan
        lum_err_for_mass_local = np.array(xray_props_local.luminosity_error, dtype=float)
        lum_err_for_mass_local[~np.isfinite(lum_err_for_mass_local) | (lum_err_for_mass_local < 0)] = np.nan

        mass_from_lum_local = estimate_mass_from_luminosity(
            luminosity=lum_for_mass_local,
            luminosity_error=lum_err_for_mass_local,
            redshift=redshift_subset,
            scaling_relation=lum_scaling,
            cosmology=cosmology,
            verbose=verbose
        )

        return {
            'phot_result': phot,
            'det_result': det_res,
            'upper_limits': upper_limits_local,
            'flux': flux_local,
            'flux_error': flux_error_local,
            'xray_props': xray_props_local,
            'flux_upper_limits': flux_upper_limits_local,
            'flux_upper_error': flux_upper_error_local,
            'xray_props_upper': xray_props_upper_local,
            'temperature': temperature_local,
            'temperature_error': temperature_error_local,
            'mass_from_temp': mass_from_temp_local,
            'mass_from_lum': mass_from_lum_local,
        }

    # Run photometry with redshift-based map selection
    if use_redshift_selection:
        # Split groups by redshift and process separately
        low_z_mask = use_full_map
        high_z_mask = use_masked_map
        
        # Process low-z groups with full map
        if np.any(low_z_mask):
            logger.info("Processing %d low-z groups with full map", int(np.sum(low_z_mask)))
            analysis_low_z = run_photometry_sequence(
                xray_maps['full'],
                ra[low_z_mask],
                dec[low_z_mask],
                redshift[low_z_mask],
                aperture_arcsec[low_z_mask],
                background_inner_arcsec[low_z_mask],
                background_outer_arcsec[low_z_mask]
            )
        else:
            analysis_low_z = None
        
        # Process high-z groups with masked map
        if np.any(high_z_mask):
            logger.info("Processing %d high-z groups with masked map", int(np.sum(high_z_mask)))
            analysis_high_z = run_photometry_sequence(
                xray_maps['masked'],
                ra[high_z_mask],
                dec[high_z_mask],
                redshift[high_z_mask],
                aperture_arcsec[high_z_mask],
                background_inner_arcsec[high_z_mask],
                background_outer_arcsec[high_z_mask]
            )
        else:
            analysis_high_z = None
        
        # Combine results
        def combine_results(result_low_z, result_high_z, low_z_mask, high_z_mask, n_total):
            """Combine results from low-z and high-z processing."""
            from xray_analysis.photometry import PhotometryResult
            from xray_analysis.detection import DetectionResult
            from xray_analysis.xray_properties import XrayProperties
            
            combined = {}
            all_keys = set()
            if result_low_z:
                all_keys.update(result_low_z.keys())
            if result_high_z:
                all_keys.update(result_high_z.keys())
            
            # Get indices for proper ordering
            low_indices = np.where(low_z_mask)[0]
            high_indices = np.where(high_z_mask)[0]
            all_indices = np.concatenate([low_indices, high_indices])
            order = np.argsort(all_indices)
            
            for key in all_keys:
                low_val = result_low_z.get(key) if result_low_z else None
                high_val = result_high_z.get(key) if result_high_z else None
                
                if low_val is not None and high_val is not None:
                    # Both present - combine arrays or objects
                    if isinstance(low_val, np.ndarray) and isinstance(high_val, np.ndarray):
                        # Combine arrays
                        dtype = low_val.dtype if low_val.dtype == high_val.dtype else float
                        combined_val = np.full(n_total, np.nan, dtype=dtype)
                        combined_val[low_z_mask] = low_val
                        combined_val[high_z_mask] = high_val
                        combined[key] = combined_val
                    elif isinstance(low_val, PhotometryResult) and isinstance(high_val, PhotometryResult):
                        # Reconstruct PhotometryResult from combined arrays in correct order
                        # Combine arrays
                        combined_source_counts = np.concatenate([low_val.source_counts, high_val.source_counts])[order]
                        combined_source_error = np.concatenate([low_val.source_error, high_val.source_error])[order]
                        combined_background = np.concatenate([low_val.background, high_val.background])[order]
                        combined_background_error = np.concatenate([low_val.background_error, high_val.background_error])[order]
                        combined_net_counts = np.concatenate([low_val.net_counts, high_val.net_counts])[order]
                        combined_net_error = np.concatenate([low_val.net_error, high_val.net_error])[order]
                        combined_snr = np.concatenate([low_val.snr, high_val.snr])[order]
                        combined_aperture_area = np.concatenate([low_val.aperture_area, high_val.aperture_area])[order]
                        combined_coverage_fraction = np.concatenate([low_val.coverage_fraction, high_val.coverage_fraction])[order]
                        combined_bg_valid_pixels = np.concatenate([low_val.background_valid_pixels, high_val.background_valid_pixels])[order]
                        
                        combined[key] = PhotometryResult(
                            source_counts=combined_source_counts,
                            source_error=combined_source_error,
                            background=combined_background,
                            background_error=combined_background_error,
                            net_counts=combined_net_counts,
                            net_error=combined_net_error,
                            snr=combined_snr,
                            aperture_area=combined_aperture_area,
                            coverage_fraction=combined_coverage_fraction,
                            background_valid_pixels=combined_bg_valid_pixels
                        )
                    elif isinstance(low_val, DetectionResult) and isinstance(high_val, DetectionResult):
                        # Reconstruct DetectionResult from combined arrays
                        combined_det = DetectionResult(
                            is_detected=np.concatenate([low_val.is_detected, high_val.is_detected])[order],
                            significance=np.concatenate([low_val.significance, high_val.significance])[order],
                            p_value=np.concatenate([low_val.p_value, high_val.p_value])[order],
                            snr=np.concatenate([low_val.snr, high_val.snr])[order]
                        )
                        combined[key] = combined_det
                    elif isinstance(low_val, XrayProperties) and isinstance(high_val, XrayProperties):
                        # Reconstruct XrayProperties from combined arrays
                        combined[key] = XrayProperties(
                            flux=np.concatenate([low_val.flux, high_val.flux])[order],
                            flux_error=np.concatenate([low_val.flux_error, high_val.flux_error])[order],
                            luminosity=np.concatenate([low_val.luminosity, high_val.luminosity])[order],
                            luminosity_error=np.concatenate([low_val.luminosity_error, high_val.luminosity_error])[order],
                            luminosity_distance=np.concatenate([low_val.luminosity_distance, high_val.luminosity_distance])[order]
                        )
                    else:
                        # Check if it's a MassEstimationResult-like object (has M200, M500, R200, R500 attributes)
                        if hasattr(low_val, 'M200') and hasattr(high_val, 'M200') and hasattr(low_val, 'method'):
                            # It's a mass result object - combine arrays
                            from xray_analysis.mass_estimation import MassEstimationResult
                            combined[key] = MassEstimationResult(
                                M200=np.concatenate([low_val.M200, high_val.M200])[order],
                                M200_err=np.concatenate([low_val.M200_err, high_val.M200_err])[order],
                                M500=np.concatenate([low_val.M500, high_val.M500])[order],
                                M500_err=np.concatenate([low_val.M500_err, high_val.M500_err])[order],
                                R200=np.concatenate([low_val.R200, high_val.R200])[order],
                                R500=np.concatenate([low_val.R500, high_val.R500])[order],
                                method=low_val.method  # Use method from low-z (should be same)
                            )
                        else:
                            # Scalar or other object - prefer low-z
                            combined[key] = low_val
                elif low_val is not None:
                    # Only low-z - expand to full size
                    if isinstance(low_val, np.ndarray):
                        combined_val = np.full(n_total, np.nan, dtype=low_val.dtype)
                        combined_val[low_z_mask] = low_val
                        combined[key] = combined_val
                    elif isinstance(low_val, XrayProperties):
                        # Expand XrayProperties to full size
                        expanded_flux = np.full(n_total, np.nan, dtype=float)
                        expanded_flux[low_z_mask] = low_val.flux
                        expanded_flux_err = np.full(n_total, np.nan, dtype=float)
                        expanded_flux_err[low_z_mask] = low_val.flux_error
                        expanded_lum = np.full(n_total, np.nan, dtype=float)
                        expanded_lum[low_z_mask] = low_val.luminosity
                        expanded_lum_err = np.full(n_total, np.nan, dtype=float)
                        expanded_lum_err[low_z_mask] = low_val.luminosity_error
                        expanded_dl = np.full(n_total, np.nan, dtype=float)
                        expanded_dl[low_z_mask] = low_val.luminosity_distance
                        combined[key] = XrayProperties(
                            flux=expanded_flux,
                            flux_error=expanded_flux_err,
                            luminosity=expanded_lum,
                            luminosity_error=expanded_lum_err,
                            luminosity_distance=expanded_dl
                        )
                    elif hasattr(low_val, 'M200') and hasattr(low_val, 'method'):
                        # Expand MassEstimationResult to full size
                        from xray_analysis.mass_estimation import MassEstimationResult
                        expanded_M200 = np.full(n_total, np.nan, dtype=float)
                        expanded_M200[low_z_mask] = low_val.M200
                        expanded_M200_err = np.full(n_total, np.nan, dtype=float)
                        expanded_M200_err[low_z_mask] = low_val.M200_err
                        expanded_M500 = np.full(n_total, np.nan, dtype=float)
                        expanded_M500[low_z_mask] = low_val.M500
                        expanded_M500_err = np.full(n_total, np.nan, dtype=float)
                        expanded_M500_err[low_z_mask] = low_val.M500_err
                        expanded_R200 = np.full(n_total, np.nan, dtype=float)
                        expanded_R200[low_z_mask] = low_val.R200
                        expanded_R500 = np.full(n_total, np.nan, dtype=float)
                        expanded_R500[low_z_mask] = low_val.R500
                        combined[key] = MassEstimationResult(
                            M200=expanded_M200,
                            M200_err=expanded_M200_err,
                            M500=expanded_M500,
                            M500_err=expanded_M500_err,
                            R200=expanded_R200,
                            R500=expanded_R500,
                            method=low_val.method
                        )
                    else:
                        combined[key] = low_val
                elif high_val is not None:
                    # Only high-z - expand to full size
                    if isinstance(high_val, np.ndarray):
                        combined_val = np.full(n_total, np.nan, dtype=high_val.dtype)
                        combined_val[high_z_mask] = high_val
                        combined[key] = combined_val
                    elif isinstance(high_val, XrayProperties):
                        # Expand XrayProperties to full size
                        expanded_flux = np.full(n_total, np.nan, dtype=float)
                        expanded_flux[high_z_mask] = high_val.flux
                        expanded_flux_err = np.full(n_total, np.nan, dtype=float)
                        expanded_flux_err[high_z_mask] = high_val.flux_error
                        expanded_lum = np.full(n_total, np.nan, dtype=float)
                        expanded_lum[high_z_mask] = high_val.luminosity
                        expanded_lum_err = np.full(n_total, np.nan, dtype=float)
                        expanded_lum_err[high_z_mask] = high_val.luminosity_error
                        expanded_dl = np.full(n_total, np.nan, dtype=float)
                        expanded_dl[high_z_mask] = high_val.luminosity_distance
                        combined[key] = XrayProperties(
                            flux=expanded_flux,
                            flux_error=expanded_flux_err,
                            luminosity=expanded_lum,
                            luminosity_error=expanded_lum_err,
                            luminosity_distance=expanded_dl
                        )
                    elif hasattr(high_val, 'M200') and hasattr(high_val, 'method'):
                        # Expand MassEstimationResult to full size
                        from xray_analysis.mass_estimation import MassEstimationResult
                        expanded_M200 = np.full(n_total, np.nan, dtype=float)
                        expanded_M200[high_z_mask] = high_val.M200
                        expanded_M200_err = np.full(n_total, np.nan, dtype=float)
                        expanded_M200_err[high_z_mask] = high_val.M200_err
                        expanded_M500 = np.full(n_total, np.nan, dtype=float)
                        expanded_M500[high_z_mask] = high_val.M500
                        expanded_M500_err = np.full(n_total, np.nan, dtype=float)
                        expanded_M500_err[high_z_mask] = high_val.M500_err
                        expanded_R200 = np.full(n_total, np.nan, dtype=float)
                        expanded_R200[high_z_mask] = high_val.R200
                        expanded_R500 = np.full(n_total, np.nan, dtype=float)
                        expanded_R500[high_z_mask] = high_val.R500
                        combined[key] = MassEstimationResult(
                            M200=expanded_M200,
                            M200_err=expanded_M200_err,
                            M500=expanded_M500,
                            M500_err=expanded_M500_err,
                            R200=expanded_R200,
                            R500=expanded_R500,
                            method=high_val.method
                        )
                    else:
                        combined[key] = high_val
            return combined
        
        if analysis_low_z and analysis_high_z:
            analysis = combine_results(analysis_low_z, analysis_high_z, low_z_mask, high_z_mask, len(ra))
        elif analysis_low_z:
            analysis = analysis_low_z
        elif analysis_high_z:
            analysis = analysis_high_z
        else:
            raise ValueError("No groups to process")
    else:
        # Single map mode (backward compatible)
        xray_map = xray_maps.get('single') or xray_maps.get('full') or xray_maps.get('masked')
        analysis = run_photometry_sequence(
            xray_map,
            ra,
            dec,
            redshift,
            aperture_arcsec,
            background_inner_arcsec,
            background_outer_arcsec
        )

    if extent_apply_mode == 'detected' and np.any(extent_valid_mask):
        det_mask = analysis['det_result'].is_detected & extent_valid_mask
        if np.any(det_mask):
            updated_aperture_arcsec = aperture_arcsec.copy()
            updated_aperture_arcsec[det_mask] = extent_arcsec_array[det_mask]
            logger.info(
                "Extent apply mode 'detected': updating apertures for %d sources with catalog extents",
                int(np.sum(det_mask))
            )
            if min_arcsec is not None:
                updated_aperture_arcsec = np.maximum(updated_aperture_arcsec, float(min_arcsec))
            if max_arcsec is not None:
                updated_aperture_arcsec = np.minimum(updated_aperture_arcsec, float(max_arcsec))
            updated_bg_inner_arcsec = np.maximum(background_inner_arcsec, updated_aperture_arcsec + gap_arcsec)
            updated_bg_outer_arcsec = np.maximum(background_outer_arcsec, updated_bg_inner_arcsec + gap_arcsec)
            if not np.allclose(updated_aperture_arcsec, aperture_arcsec):
                aperture_arcsec = updated_aperture_arcsec
                background_inner_arcsec = updated_bg_inner_arcsec
                background_outer_arcsec = updated_bg_outer_arcsec
                # Re-run with updated apertures
                if use_redshift_selection:
                    if np.any(low_z_mask):
                        analysis_low_z = run_photometry_sequence(
                            xray_maps['full'],
                            ra[low_z_mask],
                            dec[low_z_mask],
                            redshift[low_z_mask],
                            aperture_arcsec[low_z_mask],
                            background_inner_arcsec[low_z_mask],
                            background_outer_arcsec[low_z_mask]
                        )
                    if np.any(high_z_mask):
                        analysis_high_z = run_photometry_sequence(
                            xray_maps['masked'],
                            ra[high_z_mask],
                            dec[high_z_mask],
                            redshift[high_z_mask],
                            aperture_arcsec[high_z_mask],
                            background_inner_arcsec[high_z_mask],
                            background_outer_arcsec[high_z_mask]
                        )
                    if analysis_low_z and analysis_high_z:
                        analysis = combine_results(analysis_low_z, analysis_high_z, low_z_mask, high_z_mask, len(ra))
                    elif analysis_low_z:
                        analysis = analysis_low_z
                    elif analysis_high_z:
                        analysis = analysis_high_z
                else:
                    analysis = run_photometry_sequence(
                        xray_map,
                        ra,
                        dec,
                        redshift,
                        aperture_arcsec,
                        background_inner_arcsec,
                        background_outer_arcsec
                    )

    # Refine detected groups with adaptive apertures and X-ray peak centering
    detected_mask = analysis['det_result'].is_detected
    n_detected = np.sum(detected_mask)
    
    if n_detected > 0 and config.get('photometry', {}).get('refine_detected', True):
        logger.info("Refining %d detected groups with adaptive apertures and X-ray peak centering", n_detected)
        
        # Get R500/R200 from mass estimates (prefer temperature-based, fallback to luminosity-based)
        r500_kpc = analysis['mass_from_temp'].R500.copy()
        r200_kpc = analysis['mass_from_temp'].R200.copy()
        # Fill in missing values with luminosity-based estimates
        missing_r500 = ~np.isfinite(r500_kpc) | (r500_kpc <= 0)
        missing_r200 = ~np.isfinite(r200_kpc) | (r200_kpc <= 0)
        if np.any(missing_r500):
            r500_kpc[missing_r500] = analysis['mass_from_lum'].R500[missing_r500]
        if np.any(missing_r200):
            r200_kpc[missing_r200] = analysis['mass_from_lum'].R200[missing_r200]
        
        # Convert R500/R200 to arcsec for detected groups
        detected_indices = np.where(detected_mask)[0]
        updated_aperture_arcsec = aperture_arcsec.copy()
        updated_ra = ra.copy()
        updated_dec = dec.copy()
        updated_bg_inner_arcsec = background_inner_arcsec.copy()
        updated_bg_outer_arcsec = background_outer_arcsec.copy()
        
        # Arrays ra_xray_peak and dec_xray_peak are already initialized above
        
        # Determine which X-ray map to use for each detected group
        if use_redshift_selection:
            detected_low_z = detected_mask & low_z_mask
            detected_high_z = detected_mask & high_z_mask
        else:
            detected_low_z = detected_mask
            detected_high_z = np.zeros_like(detected_mask, dtype=bool)
        
        # Optional: R500-scaled background rings for detected groups (reduces contamination in annulus)
        bg_radius_mode = config.get('photometry', {}).get('background_radius_mode', 'fixed_kpc')
        bg_inner_factor_r500 = float(config.get('photometry', {}).get('background_inner_factor_r500', 1.5))
        bg_outer_factor_r500 = float(config.get('photometry', {}).get('background_outer_factor_r500', 2.5))
        use_adaptive_bg = (bg_radius_mode == 'adaptive_r500' and bg_outer_factor_r500 > bg_inner_factor_r500)
        if use_adaptive_bg:
            logger.info(
                "Using R500-scaled background annulus for detected groups (inner=%.1f*R500, outer=%.1f*R500)",
                bg_inner_factor_r500, bg_outer_factor_r500
            )

        # Process detected groups
        for idx in detected_indices:
            # Update aperture to R500 if available, otherwise R200, otherwise keep original
            r500_kpc_val = r500_kpc[idx]
            r200_kpc_val = r200_kpc[idx]
            
            if np.isfinite(r500_kpc_val) and r500_kpc_val > 0:
                # Use R500
                target_radius_kpc = r500_kpc_val
            elif np.isfinite(r200_kpc_val) and r200_kpc_val > 0:
                # Use R200
                target_radius_kpc = r200_kpc_val
            else:
                # Keep original aperture
                continue
            
            # Convert to arcsec
            z_val = redshift[idx]
            if np.isfinite(z_val) and z_val > 0:
                da_kpc = cosmology.angular_diameter_distance(z_val).to(u.kpc).value
                factor = u.rad.to(u.arcsec)
                target_radius_arcsec = (target_radius_kpc / da_kpc) * factor
                
                # Apply min/max constraints
                if min_arcsec is not None:
                    target_radius_arcsec = max(target_radius_arcsec, min_arcsec)
                if max_arcsec is not None:
                    target_radius_arcsec = min(target_radius_arcsec, max_arcsec)
                
                # Only update aperture if significantly different (more than 10%)
                if abs(target_radius_arcsec - aperture_arcsec[idx]) / aperture_arcsec[idx] > 0.1:
                    updated_aperture_arcsec[idx] = target_radius_arcsec

                # Background annulus: fixed (just outside aperture) or R500-scaled to reduce contamination
                if use_adaptive_bg and np.isfinite(r500_kpc_val) and r500_kpc_val > 0:
                    # R500-scaled rings: inner = factor_inner * R500, outer = factor_outer * R500 (kpc -> arcsec)
                    bg_inner_kpc = bg_inner_factor_r500 * r500_kpc_val
                    bg_outer_kpc = bg_outer_factor_r500 * r500_kpc_val
                    # Ensure outer > inner and both outside aperture
                    min_bg_width_kpc = 50.0
                    if bg_outer_kpc <= bg_inner_kpc:
                        bg_outer_kpc = bg_inner_kpc + min_bg_width_kpc
                    bg_inner_arcsec_val = (bg_inner_kpc / da_kpc) * factor
                    bg_outer_arcsec_val = (bg_outer_kpc / da_kpc) * factor
                    # Enforce gap and minimum width
                    bg_inner_arcsec_val = max(bg_inner_arcsec_val, target_radius_arcsec + gap_arcsec)
                    bg_outer_arcsec_val = max(bg_outer_arcsec_val, bg_inner_arcsec_val + gap_arcsec)
                    updated_bg_inner_arcsec[idx] = bg_inner_arcsec_val
                    updated_bg_outer_arcsec[idx] = bg_outer_arcsec_val
                else:
                    # Fixed: keep annulus just outside aperture (existing behaviour)
                    updated_bg_inner_arcsec[idx] = max(background_inner_arcsec[idx], target_radius_arcsec + gap_arcsec)
                    updated_bg_outer_arcsec[idx] = max(background_outer_arcsec[idx], updated_bg_inner_arcsec[idx] + gap_arcsec)
        
        # Find X-ray peaks for detected groups (only within R500 or R200 from analysis)
        # Arrays ra_xray_peak and dec_xray_peak are already initialized above
        n_peaks_found = 0
        for idx in detected_indices:
            # Only determine peak when we have R500 or R200 from X-ray analysis (search within that radius)
            r500_kpc_val = r500_kpc[idx]
            r200_kpc_val = r200_kpc[idx]
            if np.isfinite(r500_kpc_val) and r500_kpc_val > 0:
                peak_search_radius_kpc = r500_kpc_val
            elif np.isfinite(r200_kpc_val) and r200_kpc_val > 0:
                peak_search_radius_kpc = r200_kpc_val
            else:
                continue

            z_val = redshift[idx]
            if not (np.isfinite(z_val) and z_val > 0):
                continue
            da_kpc = cosmology.angular_diameter_distance(z_val).to(u.kpc).value
            if not (np.isfinite(da_kpc) and da_kpc > 0):
                continue
            factor_arcsec = u.rad.to(u.arcsec)
            peak_search_radius_arcsec = (peak_search_radius_kpc / da_kpc) * factor_arcsec

            # Determine which map to use
            if detected_low_z[idx]:
                xray_map_for_peak = xray_maps['full']
            elif detected_high_z[idx]:
                xray_map_for_peak = xray_maps['masked']
            else:
                xray_map_for_peak = xray_maps.get('single') or xray_maps.get('full') or xray_maps.get('masked')
            
            pixel_scale_peak = xray_map_for_peak.get_pixel_scale_arcsec()
            aperture_radius_pix = peak_search_radius_arcsec / pixel_scale_peak
            
            # Convert RA/Dec to pixel coordinates
            x_pix, y_pix = xray_map_for_peak.world_to_pixel(np.array([ra[idx]]), np.array([dec[idx]]))
            initial_x = float(x_pix[0])
            initial_y = float(y_pix[0])
            
            # Find X-ray centroid (flux-weighted) within R500/R200; fallback to local peak finder if centroid fails
            peak_x, peak_y, peak_snr = find_xray_centroid(
                data=xray_map_for_peak.data,
                initial_x=initial_x,
                initial_y=initial_y,
                aperture_radius_pix=aperture_radius_pix,
                smoothing_sigma=2.0,
                min_snr=2.0,
                error_map=xray_map_for_peak.error if hasattr(xray_map_for_peak, 'error') else None
            )
            if peak_x is None or peak_y is None:
                peak_x, peak_y, peak_snr = find_xray_peak(
                    data=xray_map_for_peak.data,
                    initial_x=initial_x,
                    initial_y=initial_y,
                    search_radius_pix=min(50.0, aperture_radius_pix),
                    smoothing_sigma=2.0,
                    min_snr=1.5,
                    error_map=xray_map_for_peak.error if hasattr(xray_map_for_peak, 'error') else None
                )
            
            if peak_x is not None and peak_y is not None:
                # Convert peak pixel coords back to world coords
                if hasattr(xray_map_for_peak, 'wcs') and xray_map_for_peak.wcs is not None:
                    try:
                        peak_coords = xray_map_for_peak.wcs.pixel_to_world(peak_x, peak_y)
                        peak_ra_deg = float(peak_coords.ra.deg)
                        peak_dec_deg = float(peak_coords.dec.deg)
                        # Store peak coordinates separately
                        ra_xray_peak[idx] = peak_ra_deg
                        dec_xray_peak[idx] = peak_dec_deg
                        # Update positions for refined photometry
                        updated_ra[idx] = peak_ra_deg
                        updated_dec[idx] = peak_dec_deg
                        n_peaks_found += 1
                    except Exception as e:
                        logger.debug(f"Failed to convert peak pixel to world coords for group {idx}: {e}")
        
        logger.info("Found X-ray peaks for %d/%d detected groups (within R500/R200)", n_peaks_found, n_detected)
        
        # Check if any updates were made
        aperture_changed = not np.allclose(updated_aperture_arcsec, aperture_arcsec)
        position_changed = not (np.allclose(updated_ra, ra) and np.allclose(updated_dec, dec))
        
        if aperture_changed or position_changed:
            logger.info("Re-running photometry for detected groups with refined apertures/positions")
            
            # Re-run photometry only for detected groups
            if use_redshift_selection:
                # Split detected groups by redshift
                det_low_z_mask = detected_low_z
                det_high_z_mask = detected_high_z
                
                refined_analysis_low_z = None
                refined_analysis_high_z = None
                
                if np.any(det_low_z_mask):
                    refined_analysis_low_z = run_photometry_sequence(
                        xray_maps['full'],
                        updated_ra[det_low_z_mask],
                        updated_dec[det_low_z_mask],
                        redshift[det_low_z_mask],
                        updated_aperture_arcsec[det_low_z_mask],
                        updated_bg_inner_arcsec[det_low_z_mask],
                        updated_bg_outer_arcsec[det_low_z_mask]
                    )
                
                if np.any(det_high_z_mask):
                    refined_analysis_high_z = run_photometry_sequence(
                        xray_maps['masked'],
                        updated_ra[det_high_z_mask],
                        updated_dec[det_high_z_mask],
                        redshift[det_high_z_mask],
                        updated_aperture_arcsec[det_high_z_mask],
                        updated_bg_inner_arcsec[det_high_z_mask],
                        updated_bg_outer_arcsec[det_high_z_mask]
                    )
                
                # Map detected groups back to their positions in the full array
                # refined_analysis_low_z contains results ONLY for detected low-z groups
                # We need to map these back to their original positions
                low_z_indices = np.where(low_z_mask)[0]  # All low-z group indices in full array
                det_low_z_indices = np.where(det_low_z_mask)[0]  # Detected low-z group indices in full array
                det_low_z_positions_in_subset = np.array([np.where(low_z_indices == idx)[0][0] for idx in det_low_z_indices])
                
                high_z_indices = np.where(high_z_mask)[0]  # All high-z group indices in full array
                det_high_z_indices = np.where(det_high_z_mask)[0]  # Detected high-z group indices in full array
                det_high_z_positions_in_subset = np.array([np.where(high_z_indices == idx)[0][0] for idx in det_high_z_indices])
                
                # Update analysis results for detected groups
                if refined_analysis_low_z is not None and len(det_low_z_indices) > 0:
                    # Update low-z detected groups
                    for key in ['phot_result', 'det_result', 'flux', 'flux_error', 'xray_props', 
                               'flux_upper_limits', 'flux_upper_error', 'xray_props_upper',
                               'temperature', 'temperature_error', 'mass_from_temp', 'mass_from_lum']:
                        if key in refined_analysis_low_z:
                            # Update only detected groups
                            if hasattr(analysis[key], '__len__') and len(analysis[key]) == len(ra):
                                if isinstance(analysis[key], np.ndarray):
                                    # Map from refined_analysis (detected groups only) to full array positions
                                    analysis[key][det_low_z_indices] = refined_analysis_low_z[key]
                                else:
                                    # For custom objects, update attributes
                                    for attr in ['source_counts', 'source_error', 'background', 'net_counts', 
                                                'net_error', 'snr', 'coverage_fraction', 'background_valid_pixels']:
                                        if hasattr(analysis[key], attr):
                                            orig_attr = getattr(analysis[key], attr)
                                            new_attr = getattr(refined_analysis_low_z[key], attr)
                                            if isinstance(orig_attr, np.ndarray):
                                                orig_attr[det_low_z_indices] = new_attr
                
                if refined_analysis_high_z is not None and len(det_high_z_indices) > 0:
                    # Update high-z detected groups
                    for key in ['phot_result', 'det_result', 'flux', 'flux_error', 'xray_props',
                               'flux_upper_limits', 'flux_upper_error', 'xray_props_upper',
                               'temperature', 'temperature_error', 'mass_from_temp', 'mass_from_lum']:
                        if key in refined_analysis_high_z:
                            if hasattr(analysis[key], '__len__') and len(analysis[key]) == len(ra):
                                if isinstance(analysis[key], np.ndarray):
                                    # Map from refined_analysis (detected groups only) to full array positions
                                    analysis[key][det_high_z_indices] = refined_analysis_high_z[key]
                                else:
                                    # For custom objects, update attributes
                                    for attr in ['source_counts', 'source_error', 'background', 'net_counts',
                                                'net_error', 'snr', 'coverage_fraction', 'background_valid_pixels']:
                                        if hasattr(analysis[key], attr):
                                            orig_attr = getattr(analysis[key], attr)
                                            new_attr = getattr(refined_analysis_high_z[key], attr)
                                            if isinstance(orig_attr, np.ndarray):
                                                orig_attr[det_high_z_indices] = new_attr
            else:
                # Single map mode
                xray_map = xray_maps.get('single') or xray_maps.get('full') or xray_maps.get('masked')
                refined_analysis = run_photometry_sequence(
                    xray_map,
                    updated_ra[detected_mask],
                    updated_dec[detected_mask],
                    redshift[detected_mask],
                    updated_aperture_arcsec[detected_mask],
                    updated_bg_inner_arcsec[detected_mask],
                    updated_bg_outer_arcsec[detected_mask]
                )
                
                # Update analysis results for detected groups
                for key in ['phot_result', 'det_result', 'flux', 'flux_error', 'xray_props',
                           'flux_upper_limits', 'flux_upper_error', 'xray_props_upper',
                           'temperature', 'temperature_error', 'mass_from_temp', 'mass_from_lum']:
                    if key in refined_analysis:
                        if hasattr(analysis[key], '__len__') and len(analysis[key]) == len(ra):
                            if isinstance(analysis[key], np.ndarray):
                                analysis[key][detected_mask] = refined_analysis[key]
                            else:
                                # For custom objects, update attributes
                                for attr in ['source_counts', 'source_error', 'background', 'net_counts',
                                            'net_error', 'snr', 'coverage_fraction', 'background_valid_pixels']:
                                    if hasattr(analysis[key], attr):
                                        orig_attr = getattr(analysis[key], attr)
                                        new_attr = getattr(refined_analysis[key], attr)
                                        if isinstance(orig_attr, np.ndarray):
                                            orig_attr[detected_mask] = new_attr
            
            # Update apertures and positions
            aperture_arcsec = updated_aperture_arcsec
            background_inner_arcsec = updated_bg_inner_arcsec
            background_outer_arcsec = updated_bg_outer_arcsec
            ra = updated_ra
            dec = updated_dec
            
            # Convert updated background arcsec values back to kpc for storage in results
            # This ensures the results table reflects the adaptive background values actually used
            updated_bg_inner_kpc = aperture_info['background_inner_kpc'].copy()
            updated_bg_outer_kpc = aperture_info['background_outer_kpc'].copy()
            factor = u.rad.to(u.arcsec)
            for idx in range(len(redshift)):
                z_val = redshift[idx]
                if np.isfinite(z_val) and z_val > 0:
                    da_kpc = cosmology.angular_diameter_distance(z_val).to(u.kpc).value
                    if np.isfinite(da_kpc) and da_kpc > 0:
                        # Convert arcsec -> kpc: kpc = (arcsec / factor) * da_kpc
                        bg_inner_kpc_val = (background_inner_arcsec[idx] / factor) * da_kpc
                        bg_outer_kpc_val = (background_outer_arcsec[idx] / factor) * da_kpc
                        if np.isfinite(bg_inner_kpc_val) and bg_inner_kpc_val > 0:
                            updated_bg_inner_kpc[idx] = bg_inner_kpc_val
                        if np.isfinite(bg_outer_kpc_val) and bg_outer_kpc_val > 0:
                            updated_bg_outer_kpc[idx] = bg_outer_kpc_val
            
            # Update aperture_info with the converted kpc values
            aperture_info['background_inner_kpc'] = updated_bg_inner_kpc
            aperture_info['background_outer_kpc'] = updated_bg_outer_kpc
            
            logger.info("Refinement complete: updated apertures for %d groups, positions for %d groups",
                        np.sum(~np.isclose(aperture_arcsec, aperture_info['aperture_arcsec'])),
                        n_peaks_found)

    phot_result = analysis['phot_result']
    det_result = analysis['det_result']
    upper_limits = analysis['upper_limits']
    flux = analysis['flux']
    flux_error = analysis['flux_error']
    xray_props = analysis['xray_props']
    flux_upper_limits = analysis['flux_upper_limits']
    flux_upper_error = analysis['flux_upper_error']
    xray_props_upper = analysis['xray_props_upper']
    temperature = analysis['temperature']
    temperature_error = analysis['temperature_error']
    mass_from_temp = analysis['mass_from_temp']
    mass_from_lum = analysis['mass_from_lum']

    median_aperture_arcsec = float(np.nanmedian(aperture_arcsec))
    finite_ap_kpc = aperture_info['aperture_kpc'][np.isfinite(aperture_info['aperture_kpc'])]
    median_aperture_kpc = float(np.nanmedian(finite_ap_kpc)) if finite_ap_kpc.size else float('nan')
    median_bg_inner_arcsec = float(np.nanmedian(background_inner_arcsec))
    median_bg_outer_arcsec = float(np.nanmedian(background_outer_arcsec))
    finite_bg_inner_kpc = aperture_info['background_inner_kpc'][np.isfinite(aperture_info['background_inner_kpc'])]
    finite_bg_outer_kpc = aperture_info['background_outer_kpc'][np.isfinite(aperture_info['background_outer_kpc'])]
    median_bg_inner_kpc = float(np.nanmedian(finite_bg_inner_kpc)) if finite_bg_inner_kpc.size else float('nan')
    median_bg_outer_kpc = float(np.nanmedian(finite_bg_outer_kpc)) if finite_bg_outer_kpc.size else float('nan')

    # Redshift-binned median background: compute and add *_Binned columns (annulus-based columns stay as-is)
    binned_median_enabled = config.get('photometry', {}).get('background_use_redshift_binned_median', False)
    net_counts_binned = None
    net_error_binned = None
    background_binned = None
    snr_binned = None
    redshift_bin_index = None
    flux_binned = None
    flux_error_binned = None
    xray_props_binned = None
    if binned_median_enabled:
        logger.info("Computing redshift-binned median background and adding *_Binned columns")
        net_counts_binned, net_error_binned, background_binned, snr_binned, redshift_bin_index = compute_redshift_binned_median_background(
            phot_result, redshift, config
        )
        flux_binned, flux_error_binned = calculate_xray_flux(
            net_counts=net_counts_binned,
            net_error=net_error_binned,
            count_rate_to_flux=config['xray']['count_rate_to_flux'],
            verbose=False
        )
        xray_props_binned = calculate_xray_luminosity(
            flux=flux_binned,
            flux_error=flux_error_binned,
            redshift=redshift,
            energy_band_kev=config['xray']['energy_band_kev'],
            cosmology=cosmology,
            k_correction=True,
            verbose=False
        )

    results = Table()
    results.meta['catalog_name'] = catalog_name
    if binned_median_enabled:
        results.meta['background_techniques'] = 'annulus_and_redshift_binned_median'
    results['Catalog_Name'] = [catalog_name] * n_analyzed
    # Add Group ID if available
    if group_id is not None:
        results['Group_ID'] = group_id[:n_analyzed]  # Ensure same length as analyzed groups
    # RA/DEC = original catalog/group center from input catalog (not X-ray peak).
    # Ensures showcase plots and member overlays are centered on the group center.
    results['RA'] = ra_catalog
    results['DEC'] = dec_catalog
    # Add X-ray peak coordinates (NaN if not found)
    results['RA_xray_peak'] = ra_xray_peak
    results['Dec_xray_peak'] = dec_xray_peak
    results['Redshift'] = redshift
    results['Source_Counts'] = phot_result.source_counts
    results['Source_Error'] = phot_result.source_error
    results['Background'] = phot_result.background
    results['Net_Counts'] = phot_result.net_counts
    results['Net_Error'] = phot_result.net_error
    results['SNR'] = phot_result.snr
    results['Aperture_Coverage_Fraction'] = phot_result.coverage_fraction
    results['Background_Valid_Pixels'] = phot_result.background_valid_pixels
    results['Is_Detected'] = det_result.is_detected
    results['Significance_Sigma'] = det_result.significance
    results['P_Value'] = det_result.p_value
    results['Flux_erg_cm2_s'] = flux
    results['Flux_Error'] = flux_error
    results['Luminosity_erg_s'] = xray_props.luminosity
    results['Luminosity_Error'] = xray_props.luminosity_error
    results['Upper_Limit_Counts'] = upper_limits
    results['Upper_Limit_Flux_erg_cm2_s'] = flux_upper_limits
    results['Upper_Limit_Flux_Error'] = flux_upper_error
    results['Upper_Limit_Luminosity'] = xray_props_upper.luminosity
    results['Upper_Limit_Luminosity_Error'] = xray_props_upper.luminosity_error
    results['Temperature_keV'] = temperature
    results['Temperature_Error'] = temperature_error
    results['Luminosity_Distance_Mpc'] = xray_props.luminosity_distance
    results['Aperture_Arcsec'] = aperture_arcsec
    results['Extent_Arcsec'] = extent_arcsec_array
    results['Extent_Apply_Mode'] = [extent_apply_mode] * len(results)
    if np.any(np.isfinite(aperture_info['aperture_kpc'])):
        results['Aperture_kpc'] = aperture_info['aperture_kpc']
    results['Background_Inner_Arcsec'] = background_inner_arcsec
    results['Background_Outer_Arcsec'] = background_outer_arcsec
    if np.any(np.isfinite(aperture_info['background_inner_kpc'])):
        results['Background_Inner_kpc'] = aperture_info['background_inner_kpc']
        results['Background_Outer_kpc'] = aperture_info['background_outer_kpc']

    # Redshift-binned median background results (same run; both techniques in one table)
    if binned_median_enabled and net_counts_binned is not None:
        results['Background_Binned'] = background_binned
        results['Net_Counts_Binned'] = net_counts_binned
        results['Net_Error_Binned'] = net_error_binned
        results['SNR_Binned'] = snr_binned
        snr_threshold = float(config.get('detection', {}).get('snr_threshold', 2.0))
        results['Is_Detected_Binned'] = (np.isfinite(snr_binned) & (snr_binned >= snr_threshold))
        results['Redshift_Bin_Index'] = redshift_bin_index
        results['Flux_Binned_erg_cm2_s'] = flux_binned
        results['Flux_Error_Binned'] = flux_error_binned
        results['Luminosity_Binned_erg_s'] = xray_props_binned.luminosity
        results['Luminosity_Error_Binned'] = xray_props_binned.luminosity_error

    results['M200_Temp_Msun'] = mass_from_temp.M200
    results['M200_Temp_Error'] = mass_from_temp.M200_err
    results['M500_Temp_Msun'] = mass_from_temp.M500
    results['M500_Temp_Error'] = mass_from_temp.M500_err
    results['R200_Temp_kpc'] = mass_from_temp.R200
    results['R500_Temp_kpc'] = mass_from_temp.R500
    results['Mass_Method_Temp'] = [mass_from_temp.method] * len(results)
    results['M200_Luminosity_Msun'] = mass_from_lum.M200
    results['M200_Luminosity_Error'] = mass_from_lum.M200_err
    results['M500_Luminosity_Msun'] = mass_from_lum.M500
    results['M500_Luminosity_Error'] = mass_from_lum.M500_err
    results['R200_Luminosity_kpc'] = mass_from_lum.R200
    results['R500_Luminosity_kpc'] = mass_from_lum.R500
    results['Mass_Method_Luminosity'] = [mass_from_lum.method] * len(results)
    # For backward compatibility, keep R200_kpc and R500_kpc as temperature-based
    # (since temperature-based masses are typically more reliable)
    results['R200_kpc'] = mass_from_temp.R200
    results['R500_kpc'] = mass_from_temp.R500

    log_m200_temp, log_m200_temp_err = _compute_log10_with_error(mass_from_temp.M200, mass_from_temp.M200_err)
    log_m500_temp, log_m500_temp_err = _compute_log10_with_error(mass_from_temp.M500, mass_from_temp.M500_err)
    log_m200_lum, log_m200_lum_err = _compute_log10_with_error(mass_from_lum.M200, mass_from_lum.M200_err)
    log_m500_lum, log_m500_lum_err = _compute_log10_with_error(mass_from_lum.M500, mass_from_lum.M500_err)

    results['Log10_M200_Temp'] = log_m200_temp
    results['Log10_M200_Temp_Error'] = log_m200_temp_err
    results['Log10_M500_Temp'] = log_m500_temp
    results['Log10_M500_Temp_Error'] = log_m500_temp_err
    results['Log10_M200_Luminosity'] = log_m200_lum
    results['Log10_M200_Luminosity_Error'] = log_m200_lum_err
    results['Log10_M500_Luminosity'] = log_m500_lum
    results['Log10_M500_Luminosity_Error'] = log_m500_lum_err

    detected_mask = np.asarray(det_result.is_detected, dtype=bool)
    detected_flux = results['Flux_erg_cm2_s'][detected_mask]
    median_detected_flux = float(np.nanmedian(detected_flux)) if detected_flux.size else np.nan
    q25_detected_flux = float(np.nanpercentile(detected_flux, 25)) if detected_flux.size else np.nan

    ul_flux = np.asarray(results['Upper_Limit_Flux_erg_cm2_s'], dtype=float)
    if np.isfinite(median_detected_flux) and median_detected_flux > 0:
        ul_flux_ratio = ul_flux / median_detected_flux
    else:
        ul_flux_ratio = np.full(len(results), np.nan, dtype=float)

    low_upper_limit_mask = (
        (~detected_mask)
        & np.isfinite(ul_flux)
        & np.isfinite(q25_detected_flux)
        & (q25_detected_flux > 0)
        & (ul_flux < q25_detected_flux)
    )

    results['Upper_Limit_to_Median_Flux'] = ul_flux_ratio
    results['Is_Low_Upper_Limit'] = low_upper_limit_mask

    # Flag suspected false positives (annulus-only, marginal SNR, high-z low-Lx, or contaminated).
    # They remain in the catalog and keep all columns; only Is_Suspected_False_Positive is set.
    # Stacking can optionally exclude them via config stacking.exclude_suspected_false_positives.
    det_cfg = config.get('detection', {})
    snr_robust = float(det_cfg.get('snr_robust_min', 3.0))
    z_high_cut = float(det_cfg.get('redshift_high_z_cut', 2.0))
    lx_min_high_z = float(det_cfg.get('luminosity_min_high_z_erg_s', 1.0e43))
    flag_suspected = det_cfg.get('flag_suspected_false_positives', True)
    is_detected = np.asarray(results['Is_Detected'], dtype=bool)
    snr = np.asarray(results['SNR'], dtype=float)
    redshift_arr = np.asarray(results['Redshift'], dtype=float)
    lx = np.asarray(results['Luminosity_erg_s'], dtype=float)
    if flag_suspected and np.any(is_detected):
        # Marginal SNR: detected but SNR < snr_robust
        marginal_snr = is_detected & np.isfinite(snr) & (snr < snr_robust)
        # Annulus-only: detected with annulus but not with binned median (when binned available)
        annulus_only = np.zeros(len(results), dtype=bool)
        if 'Is_Detected_Binned' in results.colnames:
            is_det_binned = np.asarray(results['Is_Detected_Binned'], dtype=bool)
            annulus_only = is_detected & (~is_det_binned)
        # Check for projected contamination from low-z groups
        check_contamination = det_cfg.get('check_projected_contamination', False)
        is_contaminated = np.zeros(len(results), dtype=bool)
        contamination_severity = np.full(len(results), 0.0, dtype=float)
        
        if check_contamination:
            logger.info("Checking for projected contamination from low-z groups...")
            low_z_thresh = float(det_cfg.get('low_z_threshold', 1.0))
            high_z_thresh = float(det_cfg.get('high_z_threshold', 1.5))
            contam_radius_factor = float(det_cfg.get('contamination_radius_factor', 2.0))  # Reduced default
            require_elevated_bg = det_cfg.get('contamination_require_elevated_background', True)
            bg_elevation_factor = float(det_cfg.get('contamination_background_factor', 1.5))
            
            # Get background arrays if available
            bg_annulus = np.asarray(results['Background'], dtype=float) if 'Background' in results.colnames else None
            bg_binned = np.asarray(results['Background_Binned'], dtype=float) if 'Background_Binned' in results.colnames else None
            
            is_contaminated, contamination_severity, contam_info = check_projected_contamination(
                ra=ra,
                dec=dec,
                redshift=redshift_arr,
                is_detected=is_detected,
                m200=np.asarray(results['M200_Luminosity_Msun'], dtype=float) if 'M200_Luminosity_Msun' in results.colnames else None,
                r500=np.asarray(results['R500_Luminosity_kpc'], dtype=float) if 'R500_Luminosity_kpc' in results.colnames else None,
                background_annulus=bg_annulus,
                background_binned=bg_binned,
                low_z_threshold=low_z_thresh,
                high_z_threshold=high_z_thresh,
                contamination_radius_factor=contam_radius_factor,
                require_elevated_background=require_elevated_bg,
                background_elevation_factor=bg_elevation_factor,
                cosmology=cosmology
            )
            
            results['Is_Projected_Contaminated'] = is_contaminated
            results['Contamination_Severity'] = contamination_severity
            
            n_contaminated = int(np.sum(is_contaminated))
            logger.info(f"Found {n_contaminated} potentially contaminated high-z groups")
        else:
            results['Is_Projected_Contaminated'] = np.zeros(len(results), dtype=bool)
            results['Contamination_Severity'] = np.full(len(results), 0.0, dtype=float)
        
        # High-z, low-Lx: z > z_cut and Lx < Lx_min (below Lx–z trend; likely false or biased)
        high_z_low_lx = (
            is_detected
            & np.isfinite(redshift_arr)
            & (redshift_arr > z_high_cut)
            & np.isfinite(lx)
            & (lx > 0)
            & (lx < lx_min_high_z)
        )
        # Optionally include contaminated groups as suspected false positives
        flag_contaminated = det_cfg.get('flag_contaminated_detections', False)
        contamination_flag = is_contaminated if flag_contaminated else np.zeros(len(results), dtype=bool)
        
        suspected_fp = marginal_snr | annulus_only | high_z_low_lx | contamination_flag
        results['Is_Suspected_False_Positive'] = suspected_fp
        n_suspected = int(np.sum(suspected_fp))
        n_high_z_low_lx = int(np.sum(high_z_low_lx))
        if n_suspected > 0:
            logger.info(
                "Flagged %d suspected false positives (marginal SNR, annulus-only, or z>%.1f Lx<%.1e: %d)",
                n_suspected, z_high_cut, lx_min_high_z, n_high_z_low_lx
            )
    else:
        results['Is_Suspected_False_Positive'] = np.zeros(len(results), dtype=bool)

    catalog_output_fits = results_dir / 'xray_catalog.fits'
    results.write(catalog_output_fits, format='fits', overwrite=True)
    logger.info("Saved X-ray catalog: %s", catalog_output_fits)

    catalog_output_csv = results_dir / 'xray_catalog.csv'
    results.write(catalog_output_csv, format='csv', overwrite=True)
    logger.info("Saved X-ray catalog (CSV): %s", catalog_output_csv)

    detections = results[det_result.is_detected]
    det_output_path = results_dir / 'detections.csv'
    detections.write(det_output_path, format='csv', overwrite=True)
    logger.info("Saved detections catalog: %s", det_output_path)

    stacking_result = None
    if config['stacking']['perform_stacking']:
        from xray_analysis.stacking import StackingResult as StackingResultClass

        # Optionally exclude suspected false positives and/or contaminated groups from stacking
        exclude_suspected = config['stacking'].get('exclude_suspected_false_positives', False)
        exclude_contaminated = config['stacking'].get('exclude_contaminated', False)
        
        stacking_ok = np.ones(len(results), dtype=bool)
        n_excluded_total = 0
        
        if exclude_suspected and 'Is_Suspected_False_Positive' in results.colnames:
            suspected_mask = np.asarray(results['Is_Suspected_False_Positive'], dtype=bool)
            stacking_ok &= ~suspected_mask
            n_excluded = int(np.sum(suspected_mask))
            n_excluded_total += n_excluded
            if n_excluded > 0:
                logger.info("Excluding %d suspected false positives from stacking", n_excluded)
        
        if exclude_contaminated and 'Is_Projected_Contaminated' in results.colnames:
            contaminated_mask = np.asarray(results['Is_Projected_Contaminated'], dtype=bool)
            stacking_ok &= ~contaminated_mask
            n_excluded = int(np.sum(contaminated_mask))
            n_excluded_total += n_excluded
            if n_excluded > 0:
                logger.info("Excluding %d projected-contaminated groups from stacking", n_excluded)
        
        if n_excluded_total > 0:
            logger.info("Total groups excluded from stacking: %d", n_excluded_total)
        
        ra_stacking = ra[stacking_ok]
        dec_stacking = dec[stacking_ok]
        redshift_stacking = redshift[stacking_ok]
        aperture_arcsec_stacking = aperture_arcsec[stacking_ok] if isinstance(aperture_arcsec, np.ndarray) else aperture_arcsec
        median_aperture_arcsec_stacking = float(np.nanmedian(aperture_arcsec_stacking)) if isinstance(aperture_arcsec_stacking, np.ndarray) else median_aperture_arcsec

        def _run_stacking(background_method: str):
            """Run stacking with given background_method; returns combined StackingResult or None."""
            out = None
            if use_redshift_selection:
                redshift_bins = np.array(config['stacking']['redshift_bins'])
                threshold = redshift_threshold
                low_z_mask_stacking = redshift_stacking < threshold
                stacking_low_z = None
                if np.any(low_z_mask_stacking):
                    low_z_bins = redshift_bins[redshift_bins < threshold]
                    if len(low_z_bins) > 1:
                        low_z_bins = np.append(low_z_bins, threshold)
                        stacking_low_z = stack_by_redshift(
                            xray_map=xray_maps['full'],
                            ra=ra_stacking[low_z_mask_stacking],
                            dec=dec_stacking[low_z_mask_stacking],
                            redshift=redshift_stacking[low_z_mask_stacking],
                            redshift_bins=low_z_bins,
                            aperture_radius=aperture_arcsec_stacking[low_z_mask_stacking] if isinstance(aperture_arcsec_stacking, np.ndarray) else median_aperture_arcsec_stacking,
                            min_sources_per_bin=config['stacking']['min_groups_per_bin'],
                            method=config['stacking']['stacking_method'],
                            sigma_clip=config['stacking']['sigma_clip'],
                            bootstrap_iterations=config['stacking']['bootstrap_iterations'],
                            background_inner_factor=config['stacking'].get('background_inner_factor', 1.5),
                            background_outer_factor=config['stacking'].get('background_outer_factor', 3.0),
                            background_method=background_method,
                            verbose=verbose
                        )
                high_z_mask_stacking = redshift_stacking >= threshold
                stacking_high_z = None
                if np.any(high_z_mask_stacking):
                    high_z_bins = redshift_bins[redshift_bins >= threshold]
                    if len(high_z_bins) > 1:
                        if len(high_z_bins) == 0 or high_z_bins[0] != threshold:
                            high_z_bins = np.append(threshold, high_z_bins)
                        stacking_high_z = stack_by_redshift(
                            xray_map=xray_maps['masked'],
                            ra=ra_stacking[high_z_mask_stacking],
                            dec=dec_stacking[high_z_mask_stacking],
                            redshift=redshift_stacking[high_z_mask_stacking],
                            redshift_bins=high_z_bins,
                            aperture_radius=aperture_arcsec_stacking[high_z_mask_stacking] if isinstance(aperture_arcsec_stacking, np.ndarray) else median_aperture_arcsec_stacking,
                            min_sources_per_bin=config['stacking']['min_groups_per_bin'],
                            method=config['stacking']['stacking_method'],
                            sigma_clip=config['stacking']['sigma_clip'],
                            bootstrap_iterations=config['stacking']['bootstrap_iterations'],
                            background_inner_factor=config['stacking'].get('background_inner_factor', 1.5),
                            background_outer_factor=config['stacking'].get('background_outer_factor', 3.0),
                            background_method=background_method,
                            verbose=verbose
                        )
                if stacking_low_z and stacking_high_z:
                    low_bin_edges = stacking_low_z.bin_edges
                    high_bin_edges = stacking_high_z.bin_edges
                    if len(high_bin_edges) > 0 and high_bin_edges[0] == threshold:
                        combined_bin_edges = np.concatenate([low_bin_edges, high_bin_edges[1:]])
                    else:
                        combined_bin_edges = np.concatenate([low_bin_edges, high_bin_edges])
                    combined_median_properties = {}
                    all_prop_keys = set(stacking_low_z.median_properties.keys()) | set(stacking_high_z.median_properties.keys())
                    for key in all_prop_keys:
                        low_val = stacking_low_z.median_properties.get(key, np.full(len(stacking_low_z.bin_centers), np.nan))
                        high_val = stacking_high_z.median_properties.get(key, np.full(len(stacking_high_z.bin_centers), np.nan))
                        combined_median_properties[key] = np.concatenate([low_val, high_val])
                    out = StackingResultClass(
                        bin_edges=combined_bin_edges,
                        bin_centers=np.concatenate([stacking_low_z.bin_centers, stacking_high_z.bin_centers]),
                        n_sources=np.concatenate([stacking_low_z.n_sources, stacking_high_z.n_sources]),
                        stacked_signal=np.concatenate([stacking_low_z.stacked_signal, stacking_high_z.stacked_signal]),
                        stacked_error=np.concatenate([stacking_low_z.stacked_error, stacking_high_z.stacked_error]),
                        snr=np.concatenate([stacking_low_z.snr, stacking_high_z.snr]),
                        median_properties=combined_median_properties,
                        is_valid=np.concatenate([stacking_low_z.is_valid, stacking_high_z.is_valid]),
                        background_median=np.concatenate([stacking_low_z.background_median, stacking_high_z.background_median])
                    )
                elif stacking_low_z:
                    out = stacking_low_z
                elif stacking_high_z:
                    out = stacking_high_z
            else:
                xray_map = xray_maps.get('single') or xray_maps.get('full') or xray_maps.get('masked')
                out = stack_by_redshift(
            xray_map=xray_map,
                    ra=ra_stacking,
                    dec=dec_stacking,
                    redshift=redshift_stacking,
            redshift_bins=np.array(config['stacking']['redshift_bins']),
                    aperture_radius=median_aperture_arcsec_stacking,
            min_sources_per_bin=config['stacking']['min_groups_per_bin'],
            method=config['stacking']['stacking_method'],
            sigma_clip=config['stacking']['sigma_clip'],
            bootstrap_iterations=config['stacking']['bootstrap_iterations'],
            background_inner_factor=config['stacking'].get('background_inner_factor', 1.5),
            background_outer_factor=config['stacking'].get('background_outer_factor', 3.0),
                    background_method=background_method,
            verbose=verbose
        )
            return out

        primary_bg = config['stacking'].get('background_method', 'local')
        logger.info("Stacking with primary background method: %s", primary_bg)
        stacking_result = _run_stacking(primary_bg)

        if stacking_result is not None:
            stacking_table = Table(stacking_result.to_dict())
            stacking_table.meta['catalog_name'] = catalog_name
            stacking_table['Catalog_Name'] = [catalog_name] * len(stacking_table)
            stacking_output_path = stacking_dir / 'stacking_results.fits'
            stacking_table.write(stacking_output_path, format='fits', overwrite=True)
            logger.info("Saved stacking results: %s", stacking_output_path)

            # Optionally run stacking with local background for comparison (when primary is bin_median)
            if config['stacking'].get('save_local_comparison', False) and primary_bg != 'local':
                logger.info("Running stacking with local background for comparison")
                stacking_result_local = _run_stacking('local')
                if stacking_result_local is not None:
                    stacking_table_local = Table(stacking_result_local.to_dict())
                    stacking_table_local.meta['catalog_name'] = catalog_name
                    stacking_table_local['Catalog_Name'] = [catalog_name] * len(stacking_table_local)
                    stacking_local_path = stacking_dir / 'stacking_results_local.fits'
                    stacking_table_local.write(stacking_local_path, format='fits', overwrite=True)
                    logger.info("Saved stacking comparison (local): %s", stacking_local_path)

    # For visualization, use full map (or single map if available)
    xray_map_viz = xray_maps.get('full') or xray_maps.get('single') or xray_maps.get('masked')
    plot_xray_map(
        xray_map=xray_map_viz,
        catalog=catalog,
        output_path=figures_dir / 'xray_map.png',
        title=f"{catalog_name} - COSMOS X-ray Map",
        show_sources=True,
        dpi=config['visualization']['dpi']
    )

    plot_detection_map(
        xray_map=xray_map_viz,
        ra=ra,
        dec=dec,
        is_detected=det_result.is_detected,
        snr=phot_result.snr,
        aperture_radius=median_aperture_arcsec,
        aperture_radii=aperture_arcsec,
        output_path=figures_dir / 'detection_map.png',
        dpi=config['visualization']['dpi']
    )

    plot_luminosity_redshift(
        luminosity=xray_props.luminosity,
        redshift=redshift,
        is_detected=det_result.is_detected,
        upper_limits=xray_props_upper.luminosity,
        flagged_upper_limits=np.asarray(results['Is_Low_Upper_Limit'], dtype=bool),
        suspected_false_positives=np.asarray(results['Is_Suspected_False_Positive'], dtype=bool),
        sample_label=catalog_name,
        output_path=figures_dir / 'luminosity_redshift.png',
        dpi=config['visualization']['dpi']
    )

    plot_diagnostic_panel(
        net_counts=phot_result.net_counts,
        snr=phot_result.snr,
        luminosity=xray_props.luminosity,
        redshift=redshift,
        is_detected=det_result.is_detected,
        output_path=figures_dir / 'diagnostics.png',
        dpi=config['visualization']['dpi']
    )

    try:
        plot_upper_limit_diagnostics(
            flux=flux,
            redshift=redshift,
            is_detected=det_result.is_detected,
            upper_limit_flux=flux_upper_limits,
            low_upper_limit_mask=np.asarray(results['Is_Low_Upper_Limit'], dtype=bool),
            output_path=figures_dir / 'upper_limit_diagnostics.png',
            dpi=config['visualization']['dpi'],
            sample_label=catalog_name,
        )
    except Exception:
        logger.exception("Failed to generate upper-limit diagnostics plot.")

    if stacking_result is not None:
        plot_stacking_results(
            stacking_result=stacking_result,
            output_path=figures_dir / 'stacking_results.png',
            dpi=config['visualization']['dpi']
        )

    n_detected = int(np.sum(det_result.is_detected))
    detection_rate = 100 * n_detected / n_analyzed if n_analyzed > 0 else 0.0
    n_low_upper_limits = int(np.sum(low_upper_limit_mask))
    summary_path = results_dir / 'summary.txt'
    summary_lines = [
        '=' * 70,
        f"X-RAY ANALYSIS SUMMARY - {catalog_name}",
        '=' * 70,
        '',
        f"Analysis date: {pd.Timestamp.now()}",
        '',
        f"Catalog file: {catalog_path}",
        f"Total groups in catalog: {n_groups}",
        f"Groups with X-ray coverage: {n_with_coverage}",
        f"Analyzed groups: {n_analyzed}",
        f"Detected sources: {n_detected} ({detection_rate:.1f}%)",
        f"Suspected false positives (excluded from stacking): {int(np.sum(results['Is_Suspected_False_Positive']))}",
        f"Non-detections: {n_analyzed - n_detected}",
        f"Low upper-limit non-detections (< Q1 detected flux): {n_low_upper_limits}",
    ]

    if np.isfinite(median_detected_flux):
        summary_lines.append(f"Median detected flux: {median_detected_flux:.3e} erg/cm^2/s")
    if np.isfinite(q25_detected_flux):
        summary_lines.append(f"Lower-quartile detected flux: {q25_detected_flux:.3e} erg/cm^2/s")

    summary_lines.append('')
    summary_lines.extend([
        f"Aperture mode: {aperture_info['aperture_mode']} (median radius {median_aperture_arcsec:.2f} arcsec"
        + (f" ≈ {median_aperture_kpc:.1f} kpc" if np.isfinite(median_aperture_kpc) else '') + ')',
        f"Background annulus (median): {median_bg_inner_arcsec:.2f}-{median_bg_outer_arcsec:.2f} arcsec"
        + (f" (≈ {median_bg_inner_kpc:.1f}-{median_bg_outer_kpc:.1f} kpc)" if np.isfinite(median_bg_inner_kpc) and np.isfinite(median_bg_outer_kpc) else ''),
        f"Detection threshold: {config['detection']['snr_threshold']:.1f} sigma",
        f"Minimum net signal: {min_signal_threshold:.3e} counts/s",
        f"Energy band: {config['xray']['energy_band_kev'][0]:.1f}-{config['xray']['energy_band_kev'][1]:.1f} keV",
    ])

    finite_m200_temp_all = mass_from_temp.M200[np.isfinite(mass_from_temp.M200) & (mass_from_temp.M200 > 0)]
    if finite_m200_temp_all.size > 0:
        summary_lines.append(f"Median log10 M200 (Temp scaling): {np.nanmedian(np.log10(finite_m200_temp_all)):.2f} dex")

    finite_m200_lum_all = mass_from_lum.M200[np.isfinite(mass_from_lum.M200) & (mass_from_lum.M200 > 0)]
    if finite_m200_lum_all.size > 0:
        summary_lines.append(f"Median log10 M200 (Lx scaling): {np.nanmedian(np.log10(finite_m200_lum_all)):.2f} dex")

    if n_detected > 0:
        det_mask = det_result.is_detected
        summary_lines.extend([
            '',
            'Detected sources statistics:',
            f"  Median SNR: {np.median(phot_result.snr[det_mask]):.2f}",
            f"  Median net counts: {np.median(phot_result.net_counts[det_mask]):.2f}",
            f"  Median flux: {np.median(flux[det_mask]):.2e} erg/cm^2/s",
            f"  Median log L_X: {np.median(np.log10(xray_props.luminosity[det_mask])):.2f} erg/s",
            f"  Median redshift: {np.median(redshift[det_mask]):.3f}",
            f"  Median temperature: {np.median(temperature[det_mask]):.2f} keV",
        ])

        det_m200_temp = mass_from_temp.M200[det_mask]
        det_m200_lum = mass_from_lum.M200[det_mask]
        if np.any(np.isfinite(det_m200_temp) & (det_m200_temp > 0)):
            summary_lines.extend([
                f"  Median log10 M200 (Temp scaling): {np.nanmedian(np.log10(det_m200_temp[np.isfinite(det_m200_temp) & (det_m200_temp > 0)])):.2f} dex",
            ])
        if np.any(np.isfinite(det_m200_lum) & (det_m200_lum > 0)):
            summary_lines.extend([
                f"  Median log10 M200 (Lx scaling): {np.nanmedian(np.log10(det_m200_lum[np.isfinite(det_m200_lum) & (det_m200_lum > 0)])):.2f} dex",
            ])

    with open(summary_path, 'w') as f:
        for line in summary_lines:
            f.write(line + "\n")

    logger.info("Summary written to %s", summary_path)
    logger.info("Results saved to %s", results_dir)
    logger.info("Figures saved to %s", figures_dir)

    det_mask = det_result.is_detected
    median_flux = float(np.median(flux[det_mask])) if np.any(det_mask) else float('nan')
    median_luminosity = (
        float(np.median(xray_props.luminosity[det_mask])) if np.any(det_mask) else float('nan')
    )
    median_temperature = (
        float(np.median(temperature[det_mask])) if np.any(det_mask) else float('nan')
    )
    det_m200_temp = mass_from_temp.M200[det_mask]
    median_m200_temp = (
        float(np.nanmedian(np.log10(det_m200_temp[np.isfinite(det_m200_temp) & (det_m200_temp > 0)])))
        if np.any(np.isfinite(det_m200_temp) & (det_m200_temp > 0)) else float('nan')
    )
    det_m200_lum = mass_from_lum.M200[det_mask]
    median_m200_lum = (
        float(np.nanmedian(np.log10(det_m200_lum[np.isfinite(det_m200_lum) & (det_m200_lum > 0)])))
        if np.any(np.isfinite(det_m200_lum) & (det_m200_lum > 0)) else float('nan')
    )

    return {
        'Catalog': catalog_name,
        'Catalog_Path': str(catalog_path),
        'Slug': slug,
        'Total_Groups': n_groups,
        'Groups_With_Coverage': n_with_coverage,
        'Analyzed_Groups': n_analyzed,
        'Detections': n_detected,
        'Detection_Rate_percent': round(detection_rate, 2),
        'Median_Flux_erg_cm2_s': median_flux,
        'Median_Luminosity_erg_s': median_luminosity,
        'Median_Temperature_keV': median_temperature,
        'Aperture_Mode': aperture_info['aperture_mode'],
        'Median_Aperture_arcsec': median_aperture_arcsec,
        'Median_Aperture_kpc': median_aperture_kpc,
        'Median_Background_Inner_arcsec': median_bg_inner_arcsec,
        'Median_Background_Outer_arcsec': median_bg_outer_arcsec,
        'Median_Background_Inner_kpc': median_bg_inner_kpc,
        'Median_Background_Outer_kpc': median_bg_outer_kpc,
        'Median_Log10_M200_Temp': median_m200_temp,
        'Median_Log10_M200_Luminosity': median_m200_lum,
        'Results_Dir': str(results_dir),
        'Figures_Dir': str(figures_dir),
        'Stacking_Dir': str(stacking_dir),
    }


def main():
    """Main analysis pipeline."""

    parser = argparse.ArgumentParser(description='X-ray analysis of galaxy groups')
    parser.add_argument('--config', type=str, default='config_refined_z.yaml',
                       help='Path to configuration file (default: config_refined_z.yaml)')
    args = parser.parse_args()

    logger.info("="*70)
    logger.info("X-RAY ANALYSIS OF GALAXY GROUPS - COSMOS SURVEY")
    logger.info("="*70)

    config = load_config(args.config)
    logger.info("Loaded configuration from: %s", args.config)

    catalog_entries = get_catalog_entries(config)
    logger.info("Catalogs to process: %s", ", ".join(entry['name'] for entry in catalog_entries))

    # Load X-ray maps (supports both single and dual map modes)
    xray_maps = load_xray_maps_from_config(config)

    metrics_list = []
    for entry in catalog_entries:
        # Get catalog-specific redshift threshold if available, otherwise use None (will use global default)
        catalog_threshold = entry.get('redshift_threshold', None)
        metrics = run_catalog_analysis(
            catalog_name=entry['name'],
            catalog_path=entry['path'],
            config=config,
            xray_maps=xray_maps,
            redshift_threshold=catalog_threshold
        )
        metrics_list.append(metrics)

    if len(metrics_list) > 1:
        write_comparison_summary(metrics_list, config)

    logger.info("Analysis complete for %d catalog(s)", len(metrics_list))


if __name__ == '__main__':
    main()
