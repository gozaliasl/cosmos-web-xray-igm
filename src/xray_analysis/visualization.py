"""
Visualization module for X-ray analysis.

Provides plotting functions for X-ray maps, detection maps, luminosity
functions, and diagnostic plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Circle, Rectangle
from matplotlib.lines import Line2D
from pathlib import Path
from astropy.visualization import (
    ZScaleInterval,
    ImageNormalize,
    LogStretch,
    LinearStretch,
    PercentileInterval,
)
from astropy.nddata import Cutout2D
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde
from astropy import units as u
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


def plot_xray_map(
    xray_map,
    catalog=None,
    output_path: Optional[str] = None,
    title: str = "X-ray Map",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = 'viridis',
    show_sources: bool = True,
    figsize: Tuple[float, float] = (12, 10),
    dpi: int = 150,
    show: bool = False
):
    """
    Plot X-ray map with optional source positions.

    Parameters
    ----------
    xray_map : XrayMap
        X-ray map object
    catalog : GroupCatalog, optional
        Source catalog to overlay
    output_path : str, optional
        Path to save figure
    title : str
        Plot title
    vmin, vmax : float, optional
        Color scale limits
    cmap : str
        Colormap name
    show_sources : bool
        Show source positions
    figsize : tuple
        Figure size
    dpi : int
        Figure resolution
    show : bool
        If True, display the plot (for Jupyter notebooks). If False, close after saving.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Get data
    data = xray_map.data

    # Set color scale
    if vmin is None or vmax is None:
        # Use ZScale for automatic scaling
        interval = ZScaleInterval()
        vmin_auto, vmax_auto = interval.get_limits(data[~np.isnan(data)])
        if vmin is None:
            vmin = vmin_auto
        if vmax is None:
            vmax = vmax_auto

    # Plot map
    im = ax.imshow(data, origin='lower', cmap=cmap,
                   vmin=vmin, vmax=vmax, aspect='equal')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Count Rate (counts/s/pixel)', fontsize=12)

    # Overlay sources
    if show_sources and catalog is not None:
        ra, dec = catalog.get_coordinates()
        x_pix, y_pix = xray_map.world_to_pixel(ra, dec)

        # Plot source positions
        ax.scatter(x_pix, y_pix, c='red', marker='x', s=50,
                  alpha=0.7, label=f'Groups (N={len(ra)})')

    ax.set_xlabel('X (pixels)', fontsize=12)
    ax.set_ylabel('Y (pixels)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    if show_sources and catalog is not None:
        ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved figure: {output_path}")

    if not show:
        plt.close()
    else:
        plt.show()


def plot_upper_limit_diagnostics(
    flux: np.ndarray,
    redshift: np.ndarray,
    is_detected: np.ndarray,
    upper_limit_flux: np.ndarray,
    low_upper_limit_mask: np.ndarray,
    output_path: Path,
    dpi: int = 150,
    sample_label: Optional[str] = None,
) -> None:
    """Visualize how upper limits compare to detected fluxes."""
    flux = np.asarray(flux, dtype=float)
    redshift = np.asarray(redshift, dtype=float)
    is_detected = np.asarray(is_detected, dtype=bool)
    upper_limit_flux = np.asarray(upper_limit_flux, dtype=float)
    low_upper_limit_mask = np.asarray(low_upper_limit_mask, dtype=bool)

    detected_flux = flux[is_detected]
    ul_flux = upper_limit_flux[~is_detected]
    flagged_ul_flux = upper_limit_flux[~is_detected & low_upper_limit_mask]

    detected_flux = detected_flux[np.isfinite(detected_flux) & (detected_flux > 0)]
    ul_flux = ul_flux[np.isfinite(ul_flux) & (ul_flux > 0)]
    flagged_ul_flux = flagged_ul_flux[np.isfinite(flagged_ul_flux) & (flagged_ul_flux > 0)]

    redshift_flagged = redshift[~is_detected & low_upper_limit_mask]
    redshift_other = redshift[~is_detected & ~low_upper_limit_mask]
    redshift_flagged = redshift_flagged[np.isfinite(redshift_flagged)]
    redshift_other = redshift_other[np.isfinite(redshift_other)]
    finite_redshift = redshift[np.isfinite(redshift)]

    if detected_flux.size == 0 and flagged_ul_flux.size == 0:
        logger.warning("No valid flux values to plot upper-limit diagnostics.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=dpi)

    ax_flux = axes[0]
    if detected_flux.size:
        ax_flux.hist(np.log10(detected_flux), bins=30, histtype='step',
                     linewidth=1.6, label='Detected flux', color='#1f77b4')
        med_flux = np.log10(np.median(detected_flux))
        q1_flux = np.log10(np.percentile(detected_flux, 25))
        ax_flux.axvline(q1_flux, color='#1f77b4', linestyle='--', linewidth=1.2, alpha=0.8,
                        label='Detected Q1')
        ax_flux.axvline(med_flux, color='#1f77b4', linestyle=':', linewidth=1.2, alpha=0.8,
                        label='Detected median')
    if ul_flux.size:
        ax_flux.hist(np.log10(ul_flux), bins=30, histtype='step',
                     linewidth=1.2, label='All upper limits', color='#ff7f0e')
    if flagged_ul_flux.size:
        ax_flux.hist(np.log10(flagged_ul_flux), bins=30, histtype='step',
                     linewidth=1.6, label='Flagged UL (< Q1 detect)', color='#d62728')
    ax_flux.set_xlabel(r'log$_{10}$ Flux [erg cm$^{-2}$ s$^{-1}$]')
    ax_flux.set_ylabel('Count')
    ax_flux.legend()

    ax_z = axes[1]
    if finite_redshift.size:
        bins = np.linspace(np.nanmin(finite_redshift), np.nanmax(finite_redshift), 25)
    else:
        bins = 10
    if redshift_other.size:
        ax_z.hist(redshift_other, bins=bins, histtype='step',
                  linewidth=1.2, label='Other upper limits', color='#ff7f0e')
    if redshift_flagged.size:
        ax_z.hist(redshift_flagged, bins=bins, histtype='step',
                  linewidth=1.6, label='Flagged UL', color='#d62728')
    ax_z.set_xlabel('Redshift')
    ax_z.set_ylabel('Count')
    ax_z.legend()

    if redshift_flagged.size:
        med_z_flagged = np.median(redshift_flagged)
        ax_z.axvline(med_z_flagged, color='#d62728', linestyle='--', linewidth=1.2, alpha=0.8)
        ylim = ax_z.get_ylim()
        ax_z.annotate(f'Median flagged z = {med_z_flagged:.2f}',
                      xy=(med_z_flagged, 0.8 * ylim[1]),
                      xytext=(5, -10), textcoords='offset points',
                      color='#d62728', fontsize=10, rotation=90,
                      va='top', ha='center')

    total_non_det = int((~is_detected).sum())
    total_flagged = int((~is_detected & low_upper_limit_mask).sum())
    frac = (total_flagged / total_non_det) if total_non_det else 0.0
    ax_z.set_title(f'Flagged ULs: {total_flagged}/{total_non_det} ({frac:.1%})')

    title = 'Upper-limit diagnostics'
    if sample_label:
        title = f'{title} ({sample_label})'
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    logger.info(f"Saved figure: {output_path}")
    plt.close(fig)


def plot_detection_map(
    xray_map,
    ra: np.ndarray,
    dec: np.ndarray,
    is_detected: np.ndarray,
    snr: Optional[np.ndarray] = None,
    aperture_radius: float = 16.0,
    aperture_radii: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (14, 10),
    dpi: int = 150,
    show: bool = False,
    snr_vmin: Optional[float] = None,
    snr_vmax: Optional[float] = None
):
    """
    Plot X-ray map with detected and non-detected sources marked.

    Parameters
    ----------
    xray_map : XrayMap
        X-ray map object
    ra, dec : np.ndarray
        Source coordinates
    is_detected : np.ndarray
        Boolean array of detections
    snr : np.ndarray, optional
        Signal-to-noise ratios for color coding
    aperture_radius : float
        Aperture radius in arcseconds
    aperture_radii : np.ndarray, optional
        Per-source aperture radii in arcseconds (overrides aperture_radius if provided)
    output_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    dpi : int
        Figure resolution
    show : bool
        If True, display the plot (for Jupyter notebooks). If False, close after saving.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Plot X-ray map
    data = xray_map.data
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(data[~np.isnan(data)])

    im = ax.imshow(data, origin='lower', cmap='jet',
                   vmin=vmin, vmax=vmax, aspect='equal', alpha=0.8)

    # Convert coordinates to pixels
    x_pix, y_pix = xray_map.world_to_pixel(ra, dec)

    # Aperture radius in pixels
    pixel_scale = xray_map.get_pixel_scale_arcsec()
    if aperture_radii is not None:
        aperture_radii = np.asarray(aperture_radii, dtype=float)
        detected_radii_arcsec = aperture_radii[is_detected]
    else:
        detected_radii_arcsec = np.full(np.sum(is_detected), aperture_radius, dtype=float)
    aperture_radius_pix_default = aperture_radius / pixel_scale

    # Plot detections
    detected_x = x_pix[is_detected]
    detected_y = y_pix[is_detected]

    if snr is not None:
        detected_snr = snr[is_detected]
        # Color code by SNR with dynamic scaling
        finite_snr = detected_snr[np.isfinite(detected_snr)]
        if snr_vmin is None:
            snr_vmin_local = float(np.nanpercentile(finite_snr, 5)) if finite_snr.size else 0.0
        else:
            snr_vmin_local = float(snr_vmin)
        if snr_vmax is None:
            snr_vmax_local = float(np.nanpercentile(finite_snr, 95)) if finite_snr.size else snr_vmin_local + 1.0
        else:
            snr_vmax_local = float(snr_vmax)
        if snr_vmax_local <= snr_vmin_local:
            snr_vmax_local = snr_vmin_local + 1.0
        scatter = ax.scatter(
            detected_x, detected_y, c=detected_snr, cmap='plasma', s=100, marker='o',
            edgecolors='white', linewidths=0.8, vmin=snr_vmin_local, vmax=snr_vmax_local,
            label='Detected', zorder=10
        )
        cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('SNR', fontsize=12)
    else:
        ax.scatter(detected_x, detected_y, c='red', s=100, marker='o',
                  edgecolors='white', linewidths=2, label='Detected', zorder=10)

    # Plot non-detections
    non_detected_x = x_pix[~is_detected]
    non_detected_y = y_pix[~is_detected]
    ax.scatter(non_detected_x, non_detected_y, c='blue', s=50, marker='x',
              alpha=0.5, label='Non-detected', zorder=9)

    # Draw apertures on a few example sources
    n_examples = min(10, np.sum(is_detected))
    for i in range(n_examples):
        if i < len(detected_x):
            radius_pix = aperture_radius_pix_default
            if i < len(detected_radii_arcsec):
                radius_pix = detected_radii_arcsec[i] / pixel_scale
            circle = Circle((detected_x[i], detected_y[i]),
                          radius_pix,
                          fill=False, edgecolor='lime',
                          linewidth=1.5, alpha=0.7)
            ax.add_patch(circle)

    ax.set_xlabel('X (pixels)', fontsize=12)
    ax.set_ylabel('Y (pixels)', fontsize=12)
    ax.set_title(f'X-ray Detections (N_det={np.sum(is_detected)}/{len(ra)})',
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved figure: {output_path}")

    if not show:
        plt.close()
    else:
        plt.show()


def plot_group_showcase(
    xray_map,
    ra: float,
    dec: float,
    redshift: float,
    cosmology,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    source_radius_kpc: float = 300.0,
    background_inner_kpc: float = 500.0,
    background_outer_kpc: float = 600.0,
    size_factor: float = 1.4,
    contour_smoothing_sigma: float = 1.2,
    dpi: int = 300,
    show: bool = False,
    find_xray_peak: bool = True,
    use_xray_center: bool = False,
    r500_kpc: Optional[float] = None,
    r200_kpc: Optional[float] = None,
    aperture_kpc_actual: Optional[float] = None,
    group_id: Optional[str] = None,
    snr: Optional[float] = None,
    cmap: str = 'viridis',
    contour_filled: bool = True,
    contour_fill_alpha: float = 0.35,
    xray_map_label: Optional[str] = None,
    contour_smoothing_for_extended: float = 2.0,
    member_ra: Optional[np.ndarray] = None,
    member_dec: Optional[np.ndarray] = None,
    bcg_ra: Optional[float] = None,
    bcg_dec: Optional[float] = None,
    show_member_density: bool = True,
    density_contour_levels: int = 5,
    density_alpha: float = 0.5,
    min_members_for_density: int = 6,
    show_xray_image: bool = False,
    xray_peak_ra: Optional[float] = None,
    xray_peak_dec: Optional[float] = None,
) -> None:
    """
    Create a zoomed X-ray image with contours and aperture/background overlays.

    Parameters
    ----------
    xray_map : XrayMap
        X-ray map object containing data, error, and WCS.
    ra, dec : float
        Sky coordinates (degrees) of the group centre (catalog position).
    redshift : float
        Group redshift used to convert physical apertures to angular sizes.
    cosmology : astropy.cosmology
        Cosmology instance for distance calculations.
    output_path : str, optional
        Destination path for the figure.
    title : str, optional
        Custom plot title; if None, a default is constructed from the radii.
    source_radius_kpc : float
        Physical radius of the source aperture (default: 300 kpc).
    background_inner_kpc : float
        Inner radius of the background annulus (default: 500 kpc).
        For low-z groups (z < 0.5), automatically increased to avoid contamination.
    background_outer_kpc : float
        Outer radius of the background annulus (default: 600 kpc).
        For low-z groups (z < 0.5), automatically increased to avoid contamination.
    size_factor : float
        Multiplier controlling the cutout size relative to the outer annulus.
    contour_smoothing_sigma : float
        Gaussian smoothing (pixels) applied before contour extraction.
    dpi : int
        Output resolution in dots per inch.
    show : bool
        If True, display figure interactively; otherwise close after saving.
    find_xray_peak : bool
        If True, find and display X-ray peak/centroid position (default: True).
    use_xray_center : bool
        If True, use X-ray peak for centering apertures instead of catalog position (default: False).
    r500_kpc : float, optional
        R500 radius from mass estimate (kpc). If provided, displayed as reference circle.
    r200_kpc : float, optional
        R200 radius from mass estimate (kpc). If provided, displayed as reference circle.
    aperture_kpc_actual : float, optional
        Actual aperture radius used in analysis (kpc). Displayed in label if different from source_radius_kpc.
    group_id : str, optional
        Group ID for legend.
    snr : float, optional
        Signal-to-noise ratio for legend.
    cmap : str
        Colormap for the X-ray image (default: 'viridis'). Options: 'viridis', 'plasma', 'inferno', 'cividis', 'hot', 'magma', etc.
    contour_filled : bool
        If True, draw filled contours (contourf) in addition to line contours (default: True).
    contour_fill_alpha : float
        Alpha for filled contour regions (default: 0.35). Only used when contour_filled is True.
    xray_map_label : str, optional
        Label for the X-ray map used (e.g. "Full (unmasked)" or map filename). Shown in legend.
    contour_smoothing_for_extended : float
        Extra smoothing (pixels) applied only for contour computation to bring out extended emission (default: 2.0). Added to contour_smoothing_sigma.
    member_ra, member_dec : array-like, optional
        RA/Dec (degrees) of group member galaxies. If provided, plotted as '+' markers.
    bcg_ra, bcg_dec : float, optional
        RA/Dec (degrees) of the most massive group member (BCG). If provided, plotted with a distinct marker (e.g. star).
    show_member_density : bool
        If True, show 2D density map of group members as filled contours (default: True).
    density_contour_levels : int
        Number of density contour levels to show (default: 5).
    density_alpha : float
        Transparency of density contours (default: 0.3).
    min_members_for_density : int
        Minimum number of members to draw 2D density map; below this, only member positions
        are shown as small red '+' markers (default: 6). Use for catalogs with few members (e.g. CW-HCG).
    show_xray_image : bool
        If False (default), plot only X-ray contours + member density + centers (no full image, no colorbar).
        If True, show full X-ray image with colorbar as before.
    xray_peak_ra, xray_peak_dec : float, optional
        Pre-computed X-ray peak coordinates (degrees). If provided, these are used
        instead of finding the peak on-the-fly. If both are provided and valid,
        find_xray_peak is ignored for peak finding (but peak will still be displayed).
    """
    if not np.isfinite(redshift) or redshift <= 0:
        raise ValueError("Valid redshift is required to plot showcase image.")

    # Adjust background annulus for low-z groups to avoid contamination
    # Low-z groups have extended emission that can contaminate nearby background
    if redshift < 0.5:
        # Increase background radii for low-z groups
        bg_scale_factor = 1.5 + (0.5 - redshift) * 2.0  # Scale more for lower z
        background_inner_kpc = background_inner_kpc * bg_scale_factor
        background_outer_kpc = background_outer_kpc * bg_scale_factor
        logger.info(f"Low-z group (z={redshift:.2f}): Scaling background annulus by {bg_scale_factor:.2f}x "
                   f"({background_inner_kpc:.0f}-{background_outer_kpc:.0f} kpc)")

    # Convert physical radii to angular sizes
    da_kpc = cosmology.angular_diameter_distance(redshift).to(u.kpc).value
    factor = u.rad.to(u.arcsec)

    def kpc_to_arcsec(kpc):
        return (kpc / da_kpc) * factor

    aperture_arcsec = kpc_to_arcsec(source_radius_kpc)
    bg_inner_arcsec = kpc_to_arcsec(background_inner_kpc)
    bg_outer_arcsec = kpc_to_arcsec(background_outer_kpc)

    pixel_scale = xray_map.get_pixel_scale_arcsec()

    # Determine cutout size (in pixels) with some padding
    # Use a more conservative size_factor to avoid extending into empty areas
    # Base size on aperture + background, but limit to reasonable extent
    max_radius_arcsec = max(bg_outer_arcsec, aperture_arcsec)
    # Reduce size_factor to avoid showing too much empty space
    # Use smaller factor for larger groups (which have larger background annuli)
    if max_radius_arcsec > 60:  # Large groups (>60 arcsec radius)
        effective_size_factor = size_factor * 0.85  # Reduce by 15%
    elif max_radius_arcsec > 40:  # Medium groups
        effective_size_factor = size_factor * 0.90  # Reduce by 10%
    else:
        effective_size_factor = size_factor  # Keep original for small groups
    
    half_size_arcsec = max_radius_arcsec * effective_size_factor
    half_size_pix = max(10, int(np.ceil(half_size_arcsec / pixel_scale)))
    cutout_size = 2 * half_size_pix + 1

    # Convert world coordinates to pixel coordinates (catalog position)
    x_pix_catalog, y_pix_catalog = xray_map.world_to_pixel(np.array([ra]), np.array([dec]))
    position_catalog = (x_pix_catalog[0], y_pix_catalog[0])
    
    # Find X-ray peak/centroid if requested
    # Use pre-computed coordinates if provided, otherwise find on-the-fly
    use_precomputed_peak = (xray_peak_ra is not None and 
                            xray_peak_dec is not None and 
                            np.isfinite(xray_peak_ra) and 
                            np.isfinite(xray_peak_dec))
    
    xray_peak_snr = None
    peak_offset_arcsec = None
    
    if use_precomputed_peak:
        # Use pre-computed X-ray peak coordinates from catalog
        logger.debug(f"Using pre-computed X-ray peak coordinates: RA={xray_peak_ra:.6f}, Dec={xray_peak_dec:.6f}")
        # Calculate offset from catalog position
        from astropy.coordinates import SkyCoord
        catalog_coord = SkyCoord(ra=ra, dec=dec, unit='deg')
        peak_coord = SkyCoord(ra=xray_peak_ra, dec=xray_peak_dec, unit='deg')
        peak_offset_arcsec = catalog_coord.separation(peak_coord).arcsec
        logger.debug(f"X-ray peak offset from catalog: {peak_offset_arcsec:.1f} arcsec")
    elif find_xray_peak:
        from .peak_finding import find_xray_centroid
        
        # Use larger search radius for low-z groups
        search_radius_pix = 50.0 if redshift >= 0.5 else 100.0
        
        peak_x, peak_y, peak_snr = find_xray_centroid(
            data=xray_map.data,
            initial_x=position_catalog[0],
            initial_y=position_catalog[1],
            aperture_radius_pix=aperture_arcsec / pixel_scale,
            smoothing_sigma=contour_smoothing_sigma,
            min_snr=2.0,
            error_map=xray_map.error if hasattr(xray_map, 'error') else None
        )
        
        if peak_x is not None and peak_y is not None:
            # Convert peak pixel coords back to world coords using WCS
            if hasattr(xray_map, 'wcs') and xray_map.wcs is not None:
                try:
                    peak_coords = xray_map.wcs.pixel_to_world(peak_x, peak_y)
                    xray_peak_ra = float(peak_coords.ra.deg)
                    xray_peak_dec = float(peak_coords.dec.deg)
                except Exception as e:
                    logger.warning(f"Failed to convert peak pixel to world coords: {e}")
                    # Fallback: estimate from pixel offset
                    offset_ra_deg = (peak_x - position_catalog[0]) * pixel_scale / 3600.0 / np.cos(np.radians(dec))
                    offset_dec_deg = (peak_y - position_catalog[1]) * pixel_scale / 3600.0
                    xray_peak_ra = ra + offset_ra_deg
                    xray_peak_dec = dec + offset_dec_deg
            else:
                # Fallback: estimate from pixel offset
                offset_ra_deg = (peak_x - position_catalog[0]) * pixel_scale / 3600.0 / np.cos(np.radians(dec))
                offset_dec_deg = (peak_y - position_catalog[1]) * pixel_scale / 3600.0
                xray_peak_ra = ra + offset_ra_deg
                xray_peak_dec = dec + offset_dec_deg
            
            xray_peak_snr = peak_snr
            
            # Calculate offset
            from astropy.coordinates import SkyCoord
            catalog_coord = SkyCoord(ra=ra, dec=dec, unit='deg')
            peak_coord = SkyCoord(ra=xray_peak_ra, dec=xray_peak_dec, unit='deg')
            peak_offset_arcsec = catalog_coord.separation(peak_coord).arcsec
            
            logger.info(f"X-ray peak found: offset = {peak_offset_arcsec:.1f} arcsec, SNR = {peak_snr:.2f}")
    
    # Use X-ray peak for centering if requested and found
    if use_xray_center and xray_peak_ra is not None:
        position = xray_map.world_to_pixel(np.array([xray_peak_ra]), np.array([xray_peak_dec]))
        position = (position[0][0], position[1][0])
        center_label = "X-ray peak"
    else:
        position = position_catalog
        center_label = "Catalog"

    # Build cutout
    cutout = Cutout2D(
        xray_map.data,
        position=position,
        size=(cutout_size, cutout_size),
        wcs=xray_map.wcs,
        mode='partial',
        fill_value=np.nan
    )

    data = cutout.data
    
    # Calculate SNR map for colorbar if error map is available
    snr_map = None
    if hasattr(xray_map, 'error') and xray_map.error is not None:
        try:
            error_cutout = Cutout2D(
                xray_map.error,
                position=position,
                size=(cutout_size, cutout_size),
                wcs=xray_map.wcs,
                mode='partial',
                fill_value=np.nan
            )
            error_data = error_cutout.data
            # Calculate SNR: signal / error
            # Avoid division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                snr_map = np.where(error_data > 0, data / error_data, 0.0)
            snr_map[~np.isfinite(snr_map)] = 0.0
            logger.debug(f"SNR map calculated: min={np.nanmin(snr_map):.2f}, max={np.nanmax(snr_map):.2f}")
        except Exception as e:
            logger.warning(f"Failed to calculate SNR map for colorbar: {e}")
            snr_map = None
    # Replace NaNs for contour computation
    data_for_contours = np.array(data, copy=True)
    finite_mask = np.isfinite(data_for_contours)
    if not np.any(finite_mask):
        raise ValueError("Cutout contains no finite data for contour plotting.")
    median_val = np.nanmedian(data_for_contours[finite_mask])
    data_for_contours[~finite_mask] = median_val
    smoothed = gaussian_filter(data_for_contours, sigma=contour_smoothing_sigma)
    # Heavier smoothing for contour levels only, to bring out extended X-ray emission
    sigma_extended = contour_smoothing_sigma + contour_smoothing_for_extended
    smoothed_for_contours = gaussian_filter(data_for_contours, sigma=sigma_extended)

    interval = PercentileInterval(95.0)
    vmin, vmax = interval.get_limits(data_for_contours)
    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LinearStretch())

    # Larger figure size for better readability and 1:1 aspect ratio
    fig = plt.figure(figsize=(14, 14), dpi=dpi)
    ny, nx = data.shape
    # Contours-only mode: use regular axes and pixel coordinates so gray, contours, and overlays are visible.
    # WCS axes can misalign pixel-drawn content with the displayed world-coordinate range for some setups.
    if show_xray_image:
        ax = fig.add_subplot(111, projection=cutout.wcs)
        im = ax.imshow(data, origin='lower', cmap=cmap, norm=norm)
        cbar = plt.colorbar(im, ax=ax, fraction=0.024, pad=0.06, shrink=0.7, aspect=25)
        cbar.set_label('Count Rate (counts s$^{-1}$ pixel$^{-1}$)', fontsize=12, fontweight='bold')
        cbar.ax.tick_params(labelsize=11)
    else:
        ax = fig.add_subplot(111)
        # White/light gray background for better contrast
        ax.set_facecolor('white')
        im_bg = ax.imshow(np.ones_like(data), origin='lower', cmap='gray', vmin=0.95, vmax=1.0,
                          zorder=0, extent=(0, nx, 0, ny))
        im_bg.set_rasterized(True)
        # Ensure 1:1 aspect ratio
        ax.set_aspect('equal')

    # Overlay SNR map if available (before contours so contours are on top)
    snr_im = None
    if snr_map is not None:
        try:
            # Define sigma levels: 0, 1, 2, 3, 4, 5, 10, 20, max
            snr_max = np.nanmax(snr_map)
            snr_levels = [0, 1, 2, 3, 4, 5, 10, 20]
            if snr_max > 20:
                snr_levels.append(snr_max)
            else:
                snr_levels.append(20)
            snr_vmax = max(snr_levels)
            
            # Create discrete colormap with easily distinguishable colors
            # Colors: dark blue, blue, cyan, green, yellow, orange, red, dark red, purple
            from matplotlib.colors import ListedColormap, BoundaryNorm
            snr_colors = ['#000080',  # Dark blue (0-1)
                          '#0000FF',  # Blue (1-2)
                          '#00FFFF',  # Cyan (2-3)
                          '#00FF00',  # Green (3-4)
                          '#FFFF00',  # Yellow (4-5)
                          '#FF8000',  # Orange (5-10)
                          '#FF0000',  # Red (10-20)
                          '#800080']  # Purple (20-max)
            
            # Create boundaries for discrete levels
            snr_boundaries = [0, 1, 2, 3, 4, 5, 10, 20, snr_vmax]
            # Ensure we have enough colors for boundaries
            if len(snr_colors) < len(snr_boundaries) - 1:
                # Extend colors if needed
                snr_colors = snr_colors + [snr_colors[-1]] * (len(snr_boundaries) - 1 - len(snr_colors))
            
            snr_cmap = ListedColormap(snr_colors[:len(snr_boundaries)-1])
            snr_norm = BoundaryNorm(snr_boundaries, snr_cmap.N)
            
            # Overlay SNR map with transparency so contours are still visible
            if show_xray_image:
                # For WCS axes, use world coordinates
                snr_im = ax.imshow(snr_map, origin='lower', cmap=snr_cmap, norm=snr_norm,
                                  alpha=0.5, zorder=6, interpolation='nearest')
            else:
                # For pixel axes, use explicit extent
                snr_im = ax.imshow(snr_map, origin='lower', cmap=snr_cmap, norm=snr_norm,
                                  alpha=0.5, zorder=6, extent=(0, nx, 0, ny), interpolation='nearest')
            logger.info(f"Overlaid SNR map with discrete colors: min={np.nanmin(snr_map):.2f}, max={np.nanmax(snr_map):.2f}")
        except Exception as e:
            logger.warning(f"Failed to overlay SNR map: {e}")

    # X-ray filled contours (group emission); use lower percentiles and smoothed map for extended emission
    try:
        smin = np.nanmin(smoothed_for_contours)
        smax = np.nanmax(smoothed_for_contours)
        if np.isfinite(smin) and np.isfinite(smax) and smax > smin:
            # Fewer contour levels for cleaner appearance: use 2 levels
            contour_percentiles = [70, 90]
            contour_levels = np.nanpercentile(smoothed_for_contours, contour_percentiles)
            contour_levels = np.unique(contour_levels[np.isfinite(contour_levels)])
            eps = max(1e-10 * (smax - smin), 1e-15)
            contour_levels = contour_levels[contour_levels > smin + eps]
            if contour_levels.size > 0:
                if contour_filled:
                    if show_xray_image:
                        # Use 'magma' colormap for X-ray contours
                        ax.contourf(smoothed_for_contours, levels=contour_levels, cmap='magma',
                                   alpha=contour_fill_alpha, zorder=7, extend='min')
                        ax.contour(smoothed_for_contours, levels=contour_levels, colors='white',
                                   linewidths=2.5, alpha=0.9, zorder=8)
                    else:
                        # Pixel coordinates: explicit grid so contours align with axes
                        # Use 'magma' colormap for X-ray contours
                        x_pix = np.arange(nx)
                        y_pix = np.arange(ny)
                        ax.contourf(x_pix, y_pix, smoothed_for_contours, levels=contour_levels, cmap='magma',
                                   alpha=0.35, zorder=7, extend='min')
                        ax.contour(x_pix, y_pix, smoothed_for_contours, levels=contour_levels, colors='white',
                                   linewidths=2.5, alpha=0.85, zorder=8)
            else:
                mid = np.nanmedian(smoothed_for_contours)
                if np.isfinite(mid) and smin < mid < smax and contour_filled:
                    if show_xray_image:
                        ax.contourf(smoothed_for_contours, levels=[mid, smax], cmap='Blues',
                                   alpha=contour_fill_alpha, zorder=7)
                    else:
                        x_pix = np.arange(nx)
                        y_pix = np.arange(ny)
                        ax.contourf(x_pix, y_pix, smoothed_for_contours, levels=[mid, smax], cmap='Blues',
                                   alpha=0.5, zorder=7)
    except Exception as exc:  # pragma: no cover
        logger.warning("Unable to draw contours: %s", exc)

    center_x = (data.shape[1] - 1) / 2.0
    center_y = (data.shape[0] - 1) / 2.0
    
    # Calculate positions in cutout coordinates
    # The cutout is centered on 'position' (which is either catalog or X-ray peak)
    # We need to find where both catalog and X-ray peak are in this cutout
    
    # Catalog position in cutout coordinates
    catalog_offset_x_pix = position_catalog[0] - position[0]
    catalog_offset_y_pix = position_catalog[1] - position[1]
    catalog_x_in_cutout = center_x + catalog_offset_x_pix
    catalog_y_in_cutout = center_y + catalog_offset_y_pix
    
    # X-ray peak position in cutout coordinates (if found)
    xray_peak_x_in_cutout = None
    xray_peak_y_in_cutout = None
    if (find_xray_peak or use_precomputed_peak) and xray_peak_ra is not None:
        try:
            # Convert X-ray peak world coords to pixel coords in full map
            peak_x_pix_full, peak_y_pix_full = xray_map.world_to_pixel(
                np.array([xray_peak_ra]), np.array([xray_peak_dec])
            )
            peak_position_full = (peak_x_pix_full[0], peak_y_pix_full[0])
            
            # Calculate offset from cutout center
            peak_offset_x_pix = peak_position_full[0] - position[0]
            peak_offset_y_pix = peak_position_full[1] - position[1]
            xray_peak_x_in_cutout = center_x + peak_offset_x_pix
            xray_peak_y_in_cutout = center_y + peak_offset_y_pix
            
            logger.debug(f"X-ray peak in cutout: ({xray_peak_x_in_cutout:.1f}, {xray_peak_y_in_cutout:.1f}), "
                        f"offset from catalog: ({peak_offset_x_pix:.1f}, {peak_offset_y_pix:.1f}) pix")
        except Exception as e:
            logger.warning(f"Failed to calculate X-ray peak position in cutout: {e}")
            xray_peak_x_in_cutout = None
            xray_peak_y_in_cutout = None

    def add_circle(radius_arcsec, color, label, center=None, linestyle='-', linewidth=2.5):
        if center is None:
            center = (center_x, center_y)
        radius_pix = radius_arcsec / pixel_scale
        circle = Circle(
            center,
            radius_pix,
            edgecolor=color,
            facecolor='none',
            linewidth=linewidth,  # Consistent linewidth for all circles
            linestyle=linestyle,
            alpha=1.0,
            label=label,
            zorder=15
        )
        ax.add_patch(circle)
        return circle

    # Determine center for physical extent circles (aperture, R500, R200)
    # Use X-ray peak if found, otherwise use catalog center
    if find_xray_peak and xray_peak_ra is not None:
        if use_xray_center:
            # Cutout is centered on X-ray peak
            physical_center = (center_x, center_y)
        else:
            # Cutout is centered on catalog, use X-ray peak position
            if xray_peak_x_in_cutout is not None and xray_peak_y_in_cutout is not None:
                physical_center = (xray_peak_x_in_cutout, xray_peak_y_in_cutout)
            else:
                physical_center = (center_x, center_y)
    else:
        # No X-ray peak found, use catalog center
        physical_center = (center_x, center_y)
    
    # Add circles centered on X-ray peak (or catalog if no peak found)
    # Consistent linewidth (2.5) for all circles, use dashed for background annulus
    # Main aperture circle
    aperture_label = f'{int(source_radius_kpc)} kpc aperture'
    if aperture_kpc_actual is not None and abs(aperture_kpc_actual - source_radius_kpc) > 1.0:
        aperture_label += f' (analysis: {int(aperture_kpc_actual)} kpc)'
    add_circle(aperture_arcsec, 'lime', aperture_label, center=physical_center, linewidth=2.5)
    
    # Add R500 and R200 reference circles if available
    # R500: use different color from X-ray contours (e.g., magenta or orange)
    if r500_kpc is not None and np.isfinite(r500_kpc) and r500_kpc > 0:
        r500_arcsec = kpc_to_arcsec(r500_kpc)
        add_circle(r500_arcsec, 'magenta', f'$R_{{500}}$ = {int(r500_kpc)} kpc', 
                   center=physical_center, linewidth=2.5)
    
    if r200_kpc is not None and np.isfinite(r200_kpc) and r200_kpc > 0:
        r200_arcsec = kpc_to_arcsec(r200_kpc)
        add_circle(r200_arcsec, 'purple', f'$R_{{200}}$ = {int(r200_kpc)} kpc', 
                   center=physical_center, linewidth=2.5)
    
    # Background annulus circles centered on catalog (where photometry was done)
    # Use dashed lines for visual hierarchy
    add_circle(bg_inner_arcsec, 'cyan', f'{int(background_inner_kpc)} kpc background (inner)',
               center=(catalog_x_in_cutout, catalog_y_in_cutout), linestyle='--', linewidth=2.5)
    add_circle(bg_outer_arcsec, 'orange', f'{int(background_outer_kpc)} kpc background (outer)',
               center=(catalog_x_in_cutout, catalog_y_in_cutout), linestyle='--', linewidth=2.5)
    
    # Always show catalog center - larger markers with edge colors for visibility
    # When cutout is centered on catalog, catalog is at center_x, center_y
    # When cutout is centered on X-ray peak, catalog is offset
    ax.plot(catalog_x_in_cutout, catalog_y_in_cutout, 'rx', 
           markersize=14, markeredgewidth=3, markeredgecolor='black', 
           markerfacecolor='red', label='Catalog center', zorder=10)
    
    # Mark X-ray peak/center if found - larger markers with edge colors for visibility
    if (find_xray_peak or use_precomputed_peak) and xray_peak_ra is not None:
        # Create label with SNR if available
        if xray_peak_snr is not None and np.isfinite(xray_peak_snr):
            xray_label = f'X-ray center (SNR={xray_peak_snr:.1f})'
        else:
            xray_label = 'X-ray center'
        
        if use_xray_center:
            # Cutout is centered on X-ray peak, so peak is at center
            ax.plot(center_x, center_y, 'g+', 
                   markersize=20, markeredgewidth=3.5, markeredgecolor='black',
                   label=xray_label, zorder=10)
            # Draw line connecting centers
            ax.plot([center_x, catalog_x_in_cutout], [center_y, catalog_y_in_cutout],
                   'r--', linewidth=1.5, alpha=0.6, zorder=9)
        else:
            # Cutout is centered on catalog, so peak is offset
            if xray_peak_x_in_cutout is not None and xray_peak_y_in_cutout is not None:
                ax.plot(xray_peak_x_in_cutout, xray_peak_y_in_cutout, 'g+', 
                       markersize=20, markeredgewidth=3.5, markeredgecolor='black',
                       label=xray_label, zorder=10)
                # Draw line connecting centers
                ax.plot([center_x, xray_peak_x_in_cutout], 
                       [center_y, xray_peak_y_in_cutout],
                       'r--', linewidth=1.5, alpha=0.6, zorder=9)
            else:
                logger.warning("X-ray peak found but could not calculate position in cutout coordinates")

    # Store member/BCG plotting info to plot at the end with highest zorder
    member_plot_info = None
    bcg_plot_info = None

    # Group membership: prepare member galaxies (+) and BCG plotting. Use pixel coords when contours-only (no WCS axes).
    if member_ra is not None and member_dec is not None:
        member_ra = np.atleast_1d(np.asarray(member_ra, dtype=float))
        member_dec = np.atleast_1d(np.asarray(member_dec, dtype=float))
        if member_ra.size == member_dec.size and member_ra.size > 0:
            try:
                # Always convert to pixel coordinates for contours-only mode; use world coords only for WCS axes
                if show_xray_image:
                    # WCS axes: use world coordinates with transform
                    world_transform = getattr(ax, 'get_transform', None)
                    if world_transform is not None:
                        try:
                            transform = world_transform('icrs')
                        except Exception:
                            transform = world_transform('world')
                        # Filter to members within cutout sky bounds
                        try:
                            footprint = cutout.wcs.calc_footprint()
                            ra_lo, ra_hi = footprint[:, 0].min(), footprint[:, 0].max()
                            dec_lo, dec_hi = footprint[:, 1].min(), footprint[:, 1].max()
                            inside = ((member_ra >= ra_lo) & (member_ra <= ra_hi) &
                                     (member_dec >= dec_lo) & (member_dec <= dec_hi))
                        except Exception:
                            inside = np.ones(member_ra.size, dtype=bool)
                        
                        if np.any(inside):
                            members_ra_plot = member_ra[inside]
                            members_dec_plot = member_dec[inside]
                            n_members = len(members_ra_plot)
                            
                            # Store for plotting at the end with highest zorder
                            member_plot_info = {
                                'x': members_ra_plot,
                                'y': members_dec_plot,
                                'n': n_members,
                                'transform': transform,
                                'use_transform': True
                            }
                
                # Pixel coordinates: always used for contours-only mode, fallback for WCS if transform fails
                # Convert RA/Dec to pixel coordinates in full map, then to cutout coordinates
                mx_full, my_full = xray_map.world_to_pixel(member_ra, member_dec)
                mx_cutout = center_x + (mx_full - position[0])
                my_cutout = center_y + (my_full - position[1])
                ny, nx = data.shape
                inside = (mx_cutout >= 0) & (mx_cutout < nx) & (my_cutout >= 0) & (my_cutout < ny)
                
                logger.debug(f"Member coordinate conversion: {len(member_ra)} total members, "
                           f"cutout size ({nx}, {ny}), center ({center_x:.1f}, {center_y:.1f}), "
                           f"position ({position[0]:.1f}, {position[1]:.1f}), "
                           f"inside cutout: {np.sum(inside)}")
                
                # Plot all members that are inside OR very close to cutout (within 10% margin)
                # This handles cases where members are slightly outside due to rounding
                margin = max(10, int(0.1 * min(nx, ny)))
                inside_expanded = ((mx_cutout >= -margin) & (mx_cutout < nx + margin) & 
                                  (my_cutout >= -margin) & (my_cutout < ny + margin))
                
                if np.any(inside_expanded):
                    members_x_plot = mx_cutout[inside_expanded]
                    members_y_plot = my_cutout[inside_expanded]
                    n_members = len(members_x_plot)
                    
                    # Clip to cutout bounds for plotting
                    members_x_plot = np.clip(members_x_plot, 0, nx - 1)
                    members_y_plot = np.clip(members_y_plot, 0, ny - 1)
                    
                    logger.info(f"Plotting {n_members} members at pixel coords: "
                               f"x range [{members_x_plot.min():.1f}, {members_x_plot.max():.1f}], "
                               f"y range [{members_y_plot.min():.1f}, {members_y_plot.max():.1f}], "
                               f"cutout size ({nx}, {ny})")
                    
                    # Store for plotting at the end with highest zorder (only if not already stored)
                    if member_plot_info is None or not show_xray_image:
                        member_plot_info = {
                            'x': members_x_plot,
                            'y': members_y_plot,
                            'n': n_members,
                            'transform': None,
                            'use_transform': False
                        }
                    
                    # Plot 2D density map of members ONLY if enabled and enough members (>= 6)
                    # For groups with < 6 members, we'll plot individual markers instead
                    logger.debug(f"Density map check: show_member_density={show_member_density}, "
                              f"n_members={n_members}, min_members_for_density={min_members_for_density}, "
                              f"show_xray_image={show_xray_image}")
                    if show_member_density and n_members >= min_members_for_density:
                        try:
                            # Use pixel coordinates for density map
                            # Filter out any NaN or infinite values
                            valid = np.isfinite(members_x_plot) & np.isfinite(members_y_plot)
                            if np.sum(valid) >= min_members_for_density:
                                x_valid = members_x_plot[valid]
                                y_valid = members_y_plot[valid]
                                
                                # Create grid for density evaluation
                                x_grid = np.linspace(0, nx - 1, nx)
                                y_grid = np.linspace(0, ny - 1, ny)
                                X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
                                
                                # Calculate KDE - use adaptive bandwidth if possible
                                try:
                                    kde = gaussian_kde(np.vstack([x_valid, y_valid]))
                                    # Evaluate density on grid
                                    positions = np.vstack([X_grid.ravel(), Y_grid.ravel()])
                                    density = kde(positions).reshape(X_grid.shape)
                                except Exception as kde_error:
                                    logger.warning(f"KDE failed, using histogram instead: {kde_error}")
                                    # Fallback: use 2D histogram with smoothing
                                    density, x_edges, y_edges = np.histogram2d(
                                        x_valid, y_valid, bins=[nx//4, ny//4], 
                                        range=[[0, nx-1], [0, ny-1]]
                                    )
                                    # Normalize and smooth
                                    density = density / np.max(density) if np.max(density) > 0 else density
                                    density = gaussian_filter(density, sigma=2.0)
                                    # Interpolate to full grid using simple method
                                    from scipy.interpolate import RectBivariateSpline
                                    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
                                    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
                                    interp = RectBivariateSpline(y_centers, x_centers, density, kx=1, ky=1)
                                    density = interp(y_grid, x_grid)
                                
                                # Normalize density to [0, 1] for contour levels
                                if np.max(density) > 0:
                                    density_norm = density / np.max(density)
                                else:
                                    density_norm = density
                                
                                # Log density statistics for debugging
                                logger.info(f"Density map stats: min={np.min(density_norm):.4f}, "
                                          f"max={np.max(density_norm):.4f}, "
                                          f"mean={np.mean(density_norm):.4f}, "
                                          f"median={np.median(density_norm):.4f}")
                                
                                # Define contour levels - use lower threshold to show more of the distribution
                                # Use percentiles of actual density values to ensure we capture the distribution
                                density_flat = density_norm.flatten()
                                density_flat = density_flat[density_flat > 0]  # Only non-zero values
                                if len(density_flat) > 0:
                                    # Use percentiles: 10th, 30th, 50th, 70th, 90th
                                    percentiles = np.percentile(density_flat, [10, 30, 50, 70, 90])
                                    levels = percentiles[:density_contour_levels]
                                    # Ensure at least one level is above zero
                                    levels = levels[levels > 0.01]
                                    if len(levels) == 0:
                                        # Fallback: use linear spacing
                                        levels = np.linspace(0.05, 0.95, density_contour_levels)
                                else:
                                    # Fallback: use linear spacing
                                    levels = np.linspace(0.05, 0.95, density_contour_levels)
                                
                                # Plot filled density contours
                                if show_xray_image:
                                    # For WCS axes, need to convert pixel grid to world coordinates
                                    # This is complex, so skip density map for WCS mode
                                    logger.debug("Skipping density map in WCS mode (complex coordinate conversion)")
                                else:
                                    # Pixel coordinates: plot density contours
                                    # Plot density map as LINE CONTOURS ONLY (not filled) - overlay on X-ray contours
                                    # Use higher alpha and thicker lines for better visibility
                                    effective_alpha = max(density_alpha, 0.8)  # Ensure minimum alpha of 0.8 for visibility
                                    # Sort levels for proper contour display
                                    levels_sorted = np.sort(levels)
                                    # Plot as line contours only (not filled) - overlay on X-ray filled contours
                                    # Use thicker linewidth for better visibility
                                    contour_lines = ax.contour(
                                        X_grid, Y_grid, density_norm,
                                        levels=levels_sorted,
                                        colors='darkred',
                                        linewidths=3.0,  # Increased from 1.5 to 3.0 for better visibility
                                        alpha=effective_alpha,
                                        zorder=9  # Above X-ray filled contours (7) but below circles (15)
                                    )
                                    logger.info(f"Plotted member density map with {len(levels)} levels "
                                              f"(range {levels[0]:.3f}-{levels[-1]:.3f}), "
                                              f"alpha={effective_alpha:.2f}, "
                                              f"{n_members} members, "
                                              f"density range: [{np.min(density_norm):.4f}, {np.max(density_norm):.4f}]")
                        except Exception as e:
                            logger.warning(f"Failed to plot member density map: {e}", exc_info=True)
                else:
                    logger.warning(f"No members inside cutout bounds (even with margin). "
                                 f"Cutout: ({nx}, {ny}), center ({center_x:.1f}, {center_y:.1f}), "
                                 f"member pixel coords range: x[{mx_cutout.min():.1f}, {mx_cutout.max():.1f}], "
                                 f"y[{my_cutout.min():.1f}, {my_cutout.max():.1f}], "
                                 f"position ({position[0]:.1f}, {position[1]:.1f})")
            except Exception as e:
                logger.warning("Failed to plot group members: %s", e)
    
    # Prepare BCG plotting info
    if bcg_ra is not None and bcg_dec is not None and np.isfinite(bcg_ra) and np.isfinite(bcg_dec):
        try:
            # Always convert BCG to pixel coordinates for contours-only mode
            if show_xray_image:
                # WCS axes: use world coordinates with transform
                world_transform = getattr(ax, 'get_transform', None)
                if world_transform is not None:
                    try:
                        transform = world_transform('icrs')
                    except Exception:
                        transform = world_transform('world')
                    bcg_plot_info = {
                        'x': [bcg_ra],
                        'y': [bcg_dec],
                        'transform': transform,
                        'use_transform': True
                    }
            
            # Pixel coordinates: always used for contours-only mode, fallback for WCS if transform fails
            if bcg_plot_info is None:
                bx_full, by_full = xray_map.world_to_pixel(np.array([bcg_ra]), np.array([bcg_dec]))
                bx_cutout = center_x + (float(bx_full[0]) - position[0])
                by_cutout = center_y + (float(by_full[0]) - position[1])
                ny, nx = data.shape
                if 0 <= bx_cutout < nx and 0 <= by_cutout < ny:
                    bcg_plot_info = {
                        'x': [bx_cutout],
                        'y': [by_cutout],
                        'transform': None,
                        'use_transform': False
                    }
        except Exception as e:
            logger.warning("Failed to prepare BCG plotting: %s", e)

    # Set axis labels and grid BEFORE plotting members/BCG at the end

    # Larger axis and tick labels for publication quality
    if show_xray_image:
        ax.set_xlabel('Right Ascension', fontsize=18, fontweight='bold')
        ax.set_ylabel('Declination', fontsize=18, fontweight='bold')
    else:
        ax.set_xlabel('Pixel x', fontsize=18, fontweight='bold')
        ax.set_ylabel('Pixel y', fontsize=18, fontweight='bold')
    ax.tick_params(labelsize=14)
    
    # Add subtle grid lines for better position reading
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.8, color='gray')
    
    # Axis limits will be set after all plotting to match cutout extent

    # Remove title completely
    # ax.set_title(title, fontsize=10)  # Commented out - no title

    # Get the current legend handles and labels BEFORE plotting members/BCG
    # (We'll add members/BCG to legend manually after plotting them)
    handles, labels = ax.get_legend_handles_labels()
    
    # Set axis limits BEFORE plotting members/BCG so they're definitely within bounds
    # Set axis limits so gray background, contours, and overlays are all visible.
    if show_xray_image and cutout.wcs is not None:
        try:
            finite_mask = np.isfinite(data) & (np.abs(data) > 1e-10)
            if np.any(finite_mask):
                rows = np.any(finite_mask, axis=1)
                cols = np.any(finite_mask, axis=0)
                if np.any(rows) and np.any(cols):
                    row_min, row_max = np.where(rows)[0][[0, -1]]
                    col_min, col_max = np.where(cols)[0][[0, -1]]
                    pad_r = max(3, int(0.05 * (row_max - row_min + 1)))
                    pad_c = max(3, int(0.05 * (col_max - col_min + 1)))
                    row_min = max(0, row_min - pad_r)
                    row_max = min(ny - 1, row_max + pad_r)
                    col_min = max(0, col_min - pad_c)
                    col_max = min(nx - 1, col_max + pad_c)
                    bl = cutout.wcs.pixel_to_world(col_min, row_min)
                    tr = cutout.wcs.pixel_to_world(col_max, row_max)
                    ra_vals = np.array([bl.ra.deg, tr.ra.deg])
                    dec_vals = np.array([bl.dec.deg, tr.dec.deg])
                    ra_min_w, ra_max_w = ra_vals.min(), ra_vals.max()
                    dec_min_w, dec_max_w = dec_vals.min(), dec_vals.max()
                    ra_margin = max((ra_max_w - ra_min_w) * 0.02, 0.0001) if ra_max_w != ra_min_w else 0.0001
                    dec_margin = max((dec_max_w - dec_min_w) * 0.02, 0.0001) if dec_max_w != dec_min_w else 0.0001
                    ax.set_xlim(ra_max_w + ra_margin, ra_min_w - ra_margin)
                    ax.set_ylim(dec_min_w - dec_margin, dec_max_w + dec_margin)
        except Exception as e:
            logger.debug("Could not set axis limits: %s", e)
    elif not show_xray_image:
        # Contours-only mode: pixel axes, set limits to full cutout
        # Ensure 1:1 aspect ratio
        # Add small margin to ensure markers at edges are visible
        margin = 2
        ax.set_xlim(-margin, nx + margin)
        ax.set_ylim(-margin, ny + margin)
        ax.set_aspect('equal')

    # NOW plot members and BCG at the very end with highest zorder so they appear on top
    # Logic: if n_members >= min_members_for_density (6), show density map ONLY (no individual markers)
    #        if n_members < 6 and n_members > 0, show individual markers ONLY (no density map)
    #        if n_members == 0, nothing to plot (group has no members)
    if member_plot_info is not None:
        n_members = member_plot_info['n']
        should_plot_individual = (n_members > 0) and (n_members < min_members_for_density)
        
        if n_members == 0:
            logger.warning(f"Group has 0 members - no member galaxies to plot")
        
        if should_plot_individual:
            # Plot individual markers for small groups (< 6 members)
            try:
                logger.info(f"Plotting {n_members} individual members (n < {min_members_for_density}) at end with zorder=200, "
                           f"use_transform={member_plot_info['use_transform']}")
                if member_plot_info['use_transform']:
                    # WCS coordinates - plot individual members as red + markers
                    ax.plot(member_plot_info['x'], member_plot_info['y'], '+', color='red', markersize=15,
                            markeredgewidth=3, markeredgecolor='darkred', 
                            label=f'Group members (N={n_members})',
                            zorder=200, transform=member_plot_info['transform'], alpha=1.0)
                else:
                    # Pixel coordinates - use plot() for better visibility (same as BCG)
                    x_unique = len(np.unique(member_plot_info['x'])) if len(member_plot_info['x']) > 0 else 0
                    y_unique = len(np.unique(member_plot_info['y'])) if len(member_plot_info['y']) > 0 else 0
                    
                    logger.info(f"Member plotting: n={n_members}, "
                               f"x={member_plot_info['x'][:3] if len(member_plot_info['x']) > 0 else 'empty'}, "
                               f"y={member_plot_info['y'][:3] if len(member_plot_info['y']) > 0 else 'empty'}, "
                               f"unique=({x_unique}, {y_unique})")
                    
                    # Plot all individual members as red + markers
                    ax.plot(member_plot_info['x'], member_plot_info['y'], '+', 
                           color='red', markersize=15, markeredgewidth=3, markeredgecolor='darkred',
                           label=f'Group members (N={n_members})', 
                           zorder=200, alpha=1.0)
                    
                    logger.info(f"Plotted {n_members} individual members using plot(): "
                               f"x range [{member_plot_info['x'].min():.1f}, {member_plot_info['x'].max():.1f}], "
                               f"y range [{member_plot_info['y'].min():.1f}, {member_plot_info['y'].max():.1f}], "
                               f"unique positions: ({x_unique}, {y_unique})")
                # Add member handle to legend manually
                handles.append(plt.Line2D([0], [0], marker='+', color='w', markerfacecolor='red', 
                                         markersize=15, markeredgecolor='darkred', markeredgewidth=3, linestyle=''))
                labels.append(f'Group members (N={n_members})')
            except Exception as e:
                logger.error("Failed to plot individual group members at end: %s", e, exc_info=True)
        else:
            # For groups with >= 6 members, density map is already plotted, just add to legend
            logger.info(f"Skipping individual member markers (n={n_members} >= {min_members_for_density}), "
                       f"density map already shown")
            # Add density map to legend - use darkred to match the contour color
            handles.append(plt.Line2D([0], [0], color='darkred', linewidth=3.0, linestyle='-'))
            labels.append(f'Member density (N={n_members})')
    else:
        logger.warning("No member_plot_info to plot - members may not have been loaded or converted")
    
    if bcg_plot_info is not None:
        try:
            if bcg_plot_info['use_transform']:
                ax.plot(bcg_plot_info['x'], bcg_plot_info['y'], '*', color='gold', markersize=16, 
                        markeredgewidth=2.5, markeredgecolor='black', 
                        zorder=101, transform=bcg_plot_info['transform'])
            else:
                ax.plot(bcg_plot_info['x'], bcg_plot_info['y'], '*', color='gold', markersize=16, 
                        markeredgewidth=2.5, markeredgecolor='black', 
                        zorder=101)
            # Add BCG handle to legend manually
            handles.append(plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', 
                                     markersize=14, markeredgecolor='black', markeredgewidth=2, linestyle=''))
            labels.append('Most massive member (BCG)')
        except Exception as e:
            logger.warning("Failed to plot BCG at end: %s", e)

    # Handles and labels were retrieved earlier, members/BCG added manually above
    
    # Add redshift, SNR, Group ID, and coordinates as separate text entries in the legend
    # Use empty handles (invisible patches) for these text-only entries
    empty_patch = Rectangle((0, 0), 0, 0, fill=False, edgecolor='none', visible=False)
    
    # Add Group ID if available (first, as it's most important identifier)
    if group_id is not None:
        handles.append(empty_patch)
        labels.append(f'Group ID = {group_id}')
    
    # Add group center coordinates
    handles.append(empty_patch)
    labels.append(f'RA = {ra:.4f}°')
    handles.append(empty_patch)
    labels.append(f'Dec = {dec:.4f}°')
    
    # Add redshift
    handles.append(empty_patch)
    labels.append(f'$z$ = {redshift:.3f}')
    
    # Add SNR if available (use X-ray peak SNR if found, otherwise use provided SNR)
    if (find_xray_peak or use_precomputed_peak) and xray_peak_snr is not None:
        handles.append(empty_patch)
        labels.append(f'SNR = {xray_peak_snr:.2f}')
    elif snr is not None and np.isfinite(snr):
        handles.append(empty_patch)
        labels.append(f'SNR = {snr:.2f}')
    
    # Add X-ray map used (so user knows which map: full/unmasked vs masked)
    if xray_map_label is not None:
        handles.append(empty_patch)
        labels.append(f'Map: {xray_map_label}')

    # Improved legend formatting: larger font, more spacing, better alignment
    ax.legend(handles, labels, loc='center right', bbox_to_anchor=(-0.12, 0.5), 
              fontsize=11, frameon=False, handletextpad=0.5, columnspacing=1.0,
              labelspacing=0.8)

    # Add SNR colorbar if SNR map is available and displayed
    if snr_im is not None:
        try:
            # Define sigma levels: 0, 1, 2, 3, 4, 5, 10, 20, max
            snr_max = np.nanmax(snr_map)
            snr_levels = [0, 1, 2, 3, 4, 5, 10, 20]
            if snr_max > 20:
                snr_levels.append(snr_max)
            else:
                snr_levels.append(20)
            
            # Add colorbar for the SNR overlay image
            cbar_snr = plt.colorbar(snr_im, ax=ax, fraction=0.024, pad=0.06, shrink=0.7, aspect=25)
            cbar_snr.set_label('SNR ($\sigma$)', fontsize=12, fontweight='bold')
            cbar_snr.ax.tick_params(labelsize=11)
            cbar_snr.set_ticks(snr_levels)
            # Format ticks: show integers for small values, 1 decimal for max if > 20
            tick_labels = [f'{int(l)}' if l < 20 else f'{l:.1f}' for l in snr_levels]
            cbar_snr.set_ticklabels(tick_labels)
            logger.info(f"Added SNR colorbar with levels: {snr_levels}")
        except Exception as e:
            logger.warning(f"Failed to add SNR colorbar: {e}")

    plt.tight_layout(rect=[0.22, 0, 0.95, 1])  # Leave more space on left for legend, right for colorbar

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=dpi)
        logger.info("Saved showcase figure: %s", output_path)

    if not show:
        plt.close(fig)
    else:
        plt.show()


def plot_luminosity_redshift(
    luminosity: np.ndarray,
    redshift: np.ndarray,
    is_detected: np.ndarray,
    upper_limits: Optional[np.ndarray] = None,
    flagged_upper_limits: Optional[np.ndarray] = None,
    suspected_false_positives: Optional[np.ndarray] = None,
    stacking_redshift: Optional[np.ndarray] = None,
    stacking_luminosity: Optional[np.ndarray] = None,
    panel_label: Optional[str] = None,
    sample_label: Optional[str] = None,
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (6, 5),
    dpi: int = 300,
    show: bool = False
):
    """
    Plot X-ray luminosity vs redshift (publication-quality, no gridlines).

    Parameters
    ----------
    luminosity : np.ndarray
        X-ray luminosity in erg/s
    redshift : np.ndarray
        Redshift array
    is_detected : np.ndarray
        Boolean array of detections
    upper_limits : np.ndarray, optional
        Upper limits for non-detections
    flagged_upper_limits : np.ndarray, optional
        Boolean mask identifying flagged upper limits (e.g., below detected Q1)
    suspected_false_positives : np.ndarray, optional
        Boolean mask for suspected false positives.
    stacking_redshift : np.ndarray, optional
        Redshift bin centres for stacked median Lx (plotted as orange squares).
    stacking_luminosity : np.ndarray, optional
        Stacked median Lx per bin (same length as stacking_redshift).
    panel_label : str, optional
        Panel label, e.g. '(a)' or '(b)', drawn at top-left.
    sample_label : str, optional
        Sample name label (e.g. 'CW-All', 'CW-HCG'), drawn at top-right.
    output_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size (single panel)
    dpi : int
        Figure resolution (default 300 for publication)
    show : bool
        If True, call plt.show() instead of closing.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    # Publication style: no gridlines, clear spines
    ax.set_axisbelow(False)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
    ax.tick_params(axis='both', which='major', labelsize=11, width=1.0, length=4)

    detected_mask = np.asarray(is_detected, dtype=bool) & np.isfinite(luminosity) & (luminosity > 0)
    if suspected_false_positives is not None:
        suspected_fp = np.asarray(suspected_false_positives, dtype=bool)
        robust_detected = detected_mask & ~suspected_fp
        suspected_detected = detected_mask & suspected_fp
    else:
        robust_detected = detected_mask
        suspected_detected = np.zeros_like(detected_mask, dtype=bool)

    if upper_limits is not None:
        upper_limits = np.asarray(upper_limits, dtype=float)
        non_detected_mask = (~detected_mask) & np.isfinite(upper_limits) & (upper_limits > 0)
        if flagged_upper_limits is not None:
            flagged_upper_limits = np.asarray(flagged_upper_limits, dtype=bool)
            flagged_mask = non_detected_mask & flagged_upper_limits
            other_mask = non_detected_mask & ~flagged_upper_limits
        else:
            flagged_mask = np.zeros_like(non_detected_mask, dtype=bool)
            other_mask = non_detected_mask

        if np.any(other_mask):
            ax.scatter(redshift[other_mask], upper_limits[other_mask],
                       s=10, c='tab:gray', alpha=0.5, marker='o',
                       label=f'Upper limits (N={np.sum(other_mask)})', zorder=4)
        if np.any(flagged_mask):
            ax.scatter(redshift[flagged_mask], upper_limits[flagged_mask],
                       s=14, c='tab:blue', alpha=0.7, marker='o',
                       label=f'Flagged upper limits (N={np.sum(flagged_mask)})', zorder=5)

        total_non_det = int(np.sum(non_detected_mask))
        flagged_total = int(np.sum(flagged_mask))
        if total_non_det > 0 and flagged_total > 0:
            frac = flagged_total / total_non_det
            summary = f'Flagged: {flagged_total}/{total_non_det} ({frac:.0%})'
            ax.text(0.98, 0.04, summary, transform=ax.transAxes,
                    ha='right', va='bottom', fontsize=9, color='tab:blue',
                    bbox=dict(facecolor='white', alpha=0.85, edgecolor='#ccc', pad=1.5))

    if np.any(robust_detected):
        ax.scatter(redshift[robust_detected], luminosity[robust_detected],
                   facecolors='none', edgecolors='tab:red', linewidths=1.0,
                   s=24, alpha=0.9, label=f'Detections (N={np.sum(robust_detected)})', zorder=10)
    if np.any(suspected_detected):
        ax.scatter(redshift[suspected_detected], luminosity[suspected_detected],
                   facecolors='none', edgecolors='tab:orange', linewidths=0.8,
                   s=20, alpha=0.85, marker='s', label=f'Suspected false positives (N={np.sum(suspected_detected)})', zorder=9)

    # Stacked median Lx per redshift bin (orange squares)
    if (stacking_redshift is not None and stacking_luminosity is not None and
            len(stacking_redshift) == len(stacking_luminosity)):
        valid = np.isfinite(stacking_redshift) & np.isfinite(stacking_luminosity) & (stacking_luminosity > 0)
        if np.any(valid):
            ax.scatter(stacking_redshift[valid], stacking_luminosity[valid],
                       s=28, c='tab:orange', alpha=0.9, marker='s', zorder=11,
                       edgecolors='black', linewidths=0.8,
                       label='Stacked median Lx')

    if panel_label:
        ax.text(0.04, 0.96, panel_label, transform=ax.transAxes,
                fontsize=13, fontweight='bold', va='top', ha='left')
    if sample_label:
        ax.text(0.96, 0.96, sample_label, transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='top', ha='right')

    ax.set_xlabel('Redshift', fontsize=12)
    ax.set_ylabel(r"$L_{\mathrm{X}}$ (0.5--2.0 keV) [erg s$^{-1}$]", fontsize=12)
    ax.set_yscale('log')
    ax.legend(loc='upper left', fontsize=10, frameon=True, fancybox=False,
              edgecolor='#ccc', framealpha=0.95)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved figure: {output_path}")

    if not show:
        plt.close()
    else:
        plt.show()


def plot_stacking_results(
    stacking_result,
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 5),
    dpi: int = 150,
    show: bool = False
):
    """
    Plot stacking analysis results.

    Parameters
    ----------
    stacking_result : StackingResult
        Stacking results object
    output_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    dpi : int
        Figure resolution
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

    bin_centers = stacking_result.bin_centers
    signal = stacking_result.stacked_signal
    error = stacking_result.stacked_error
    snr = stacking_result.snr
    n_sources = stacking_result.n_sources
    valid_mask = getattr(stacking_result, 'is_valid', ~np.isnan(signal))

    # Plot stacked signal with log scale on x-axis only
    # Separate positive and zero/negative values
    finite_signal = np.isfinite(signal)
    finite_error = np.isfinite(error)
    valid_signal = valid_mask & finite_signal & (signal > 0)
    valid_upper = valid_mask & finite_error & (~valid_signal)

    # Plot positive signals
    if np.any(valid_signal):
        ax1.errorbar(bin_centers[valid_signal], signal[valid_signal],
                    yerr=error[valid_signal],
                    fmt='o-', markersize=8, capsize=5,
                    color='red', linewidth=2, label='Stacked signal')

    # Plot zero/negative signals as upper limits
    if np.any(valid_upper):
        # For zero values, plot at a small threshold or as upper limits
        upper_limit_vals = error[valid_upper] * 3  # 3-sigma upper limit
        ax1.errorbar(bin_centers[valid_upper], upper_limit_vals,
                    yerr=upper_limit_vals*0.3,
                    fmt='v', markersize=6, capsize=3, alpha=0.5,
                    color='gray', linewidth=1, label='Upper limits (non-detections)')

    # Mark bins that did not meet minimum source requirement
    invalid_mask = ~valid_mask
    if np.any(invalid_mask):
        ax1.scatter(bin_centers[invalid_mask], np.zeros(np.sum(invalid_mask)),
                    marker='x', color='darkgray', label='Insufficient sources')

    #ax1.axhline(0, color='k', linestyle='--', alpha=0.5, linewidth=0.8)
    ax1.set_xlabel('Redshift', fontsize=12)
    ax1.set_ylabel('Stacked X-ray Signal (counts)', fontsize=12)
    ax1.set_title('Stacking Analysis', fontsize=13, fontweight='bold')
    # Use symlog for y-axis to handle zero/negative values robustly
    #finite_vals = signal[np.isfinite(signal)]
    #if finite_vals.size > 0 and np.any(finite_vals != 0):
    #    ax1.set_yscale('symlog', linthresh=1e-7)
    #    abs_max = np.nanmax(np.abs(finite_vals))
    #    if np.isfinite(abs_max) and abs_max > 0:
    #        pad = abs_max * 1.2
    #        ax1.set_ylim(-pad, pad)
    #else:
    #ax1.set_ylim(5e-7, 4e-3)
    ax1.set_yscale('log', nonpositive='clip')
    ax1.legend(fontsize=10)
    
    ax1.grid(True, alpha=0.3, which='both')

    # Plot SNR and number of sources with log scale
    ax2_twin = ax2.twinx()

    line1 = ax2.plot(bin_centers, snr, 'o-', color='blue',
                     markersize=8, linewidth=2, label='SNR')
    ax2.axhline(1.5, color='red', linestyle='--', alpha=0.6, linewidth=2.5, label='1.5σ threshold')
    ax2.axhline(2, color='orange', linestyle='--', alpha=0.5, linewidth=1.5, label='2σ threshold')
    ax2.axhline(3, color='k', linestyle='--', alpha=0.4, linewidth=1, label='3σ threshold')
    ax2.set_xlabel('Redshift', fontsize=12)
    ax2.set_ylabel('Signal-to-Noise Ratio', fontsize=12, color='blue')
    #ax2.set_xscale('log')
    ax2.tick_params(axis='y', labelcolor='blue')

    # Calculate bar width in log space
    bin_widths = np.diff(np.log10(stacking_result.bin_edges))
    bar_colors = ['lightgray' if not valid else 'gray' for valid in valid_mask]
    line2 = ax2_twin.bar(bin_centers, n_sources,
                        width=bin_widths * 0.8,
                        alpha=0.4, color=bar_colors, edgecolor='dimgray', label='N sources')
    ax2_twin.set_ylabel('Number of Sources', fontsize=12, color='gray')
    ax2_twin.tick_params(axis='y', labelcolor='gray')

    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)

    ax2.set_title('Detection Significance', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved figure: {output_path}")

    if not show:
        plt.close()
    else:
        plt.show()


def plot_diagnostic_panel(
    net_counts: np.ndarray,
    snr: np.ndarray,
    luminosity: np.ndarray,
    redshift: np.ndarray,
    is_detected: np.ndarray,
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (14, 10),
    dpi: int = 150,
    show: bool = False
):
    """
    Create multi-panel diagnostic plot.

    Parameters
    ----------
    net_counts : np.ndarray
        Net source counts
    snr : np.ndarray
        Signal-to-noise ratios
    luminosity : np.ndarray
        X-ray luminosities
    redshift : np.ndarray
        Redshifts
    is_detected : np.ndarray
        Detection flags
    output_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    dpi : int
        Figure resolution
    """
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Panel 1: Count distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(net_counts[is_detected], bins=30, alpha=0.7,
            color='red', label='Detected')
    ax1.hist(net_counts[~is_detected], bins=30, alpha=0.7,
            color='blue', label='Non-detected')
    ax1.set_xlabel('Net Counts', fontsize=11)
    ax1.set_ylabel('Number of Sources', fontsize=11)
    ax1.set_title('Count Distribution', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel 2: SNR distribution
    ax2 = fig.add_subplot(gs[0, 1])
    snr_bins = np.linspace(-5, 20, 50)
    ax2.hist(snr[is_detected], bins=snr_bins, alpha=0.7,
            color='red', label='Detected')
    ax2.hist(snr[~is_detected], bins=snr_bins, alpha=0.7,
            color='blue', label='Non-detected')
    ax2.axvline(1.5, color='red', linestyle='--', linewidth=2.5, alpha=0.6, label='1.5σ threshold')
    ax2.axvline(2, color='orange', linestyle='--', linewidth=1.5, alpha=0.5, label='2σ threshold')
    ax2.axvline(3, color='k', linestyle='--', linewidth=1, alpha=0.4, label='3σ threshold')
    ax2.set_xlabel('Signal-to-Noise Ratio', fontsize=11)
    ax2.set_ylabel('Number of Sources', fontsize=11)
    ax2.set_title('SNR Distribution', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Luminosity vs redshift
    ax3 = fig.add_subplot(gs[1, 0])
    det_mask = is_detected & (luminosity > 0)
    ax3.scatter(redshift[det_mask], luminosity[det_mask],
               c=snr[det_mask], cmap='hot', s=50, alpha=0.7,
               vmin=3, vmax=10, edgecolors='k', linewidths=0.5)
    ax3.set_xlabel('Redshift', fontsize=11)
    ax3.set_ylabel(r'$L_{\rm X}$ [erg s$^{-1}$]', fontsize=11)
    ax3.set_yscale('log')
    ax3.set_title('Luminosity vs Redshift (Detected)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Panel 4: Detection fraction vs redshift
    ax4 = fig.add_subplot(gs[1, 1])
    z_bins = np.linspace(redshift.min(), redshift.max(), 10)
    z_centers = 0.5 * (z_bins[:-1] + z_bins[1:])
    det_fraction = []
    for i in range(len(z_bins)-1):
        mask = (redshift >= z_bins[i]) & (redshift < z_bins[i+1])
        if np.sum(mask) > 0:
            frac = np.sum(is_detected[mask]) / np.sum(mask)
        else:
            frac = 0
        det_fraction.append(frac)

    ax4.bar(z_centers, det_fraction, width=np.diff(z_bins)[0]*0.8,
           color='green', alpha=0.7, edgecolor='k')
    ax4.set_xlabel('Redshift', fontsize=11)
    ax4.set_ylabel('Detection Fraction', fontsize=11)
    ax4.set_ylim(0, 1)
    ax4.set_title('Detection Rate vs Redshift', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    plt.suptitle('X-ray Analysis Diagnostics', fontsize=15, fontweight='bold', y=0.995)

    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved figure: {output_path}")

    if not show:
        plt.close()
    else:
        plt.show()


def plot_cutout(
    xray_map,
    ra: float,
    dec: float,
    size_arcsec: float = 60.0,
    aperture_radius: float = 16.0,
    title: Optional[str] = None,
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 8),
    dpi: int = 150,
    show: bool = False
):
    """
    Plot cutout around a single source.

    Parameters
    ----------
    xray_map : XrayMap
        X-ray map object
    ra, dec : float
        Source coordinates
    size_arcsec : float
        Cutout size in arcseconds
    aperture_radius : float
        Aperture radius to show
    title : str, optional
        Plot title
    output_path : str, optional
        Save path
    figsize : tuple
        Figure size
    dpi : int
        Resolution
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Convert to pixels
    x_pix, y_pix = xray_map.world_to_pixel(np.array([ra]), np.array([dec]))
    x_pix, y_pix = x_pix[0], y_pix[0]

    pixel_scale = xray_map.get_pixel_scale_arcsec()
    size_pix = int(size_arcsec / pixel_scale)
    aperture_radius_pix = aperture_radius / pixel_scale

    # Extract cutout
    xi, yi = int(np.round(x_pix)), int(np.round(y_pix))
    y_min, y_max = yi - size_pix, yi + size_pix
    x_min, x_max = xi - size_pix, xi + size_pix

    cutout = xray_map.data[y_min:y_max, x_min:x_max]

    # Plot
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(cutout[~np.isnan(cutout)])
    im = ax.imshow(cutout, origin='lower', cmap='hot', vmin=vmin, vmax=vmax)

    # Mark center and aperture
    center_x, center_y = size_pix, size_pix
    ax.plot(center_x, center_y, 'g+', markersize=20, markeredgewidth=2)

    circle = Circle((center_x, center_y), aperture_radius_pix,
                   fill=False, edgecolor='lime', linewidth=2)
    ax.add_patch(circle)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if title is None:
        title = f'Source at RA={ra:.4f}, Dec={dec:.4f}'

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('X (pixels)', fontsize=10)
    ax.set_ylabel('Y (pixels)', fontsize=10)

    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved figure: {output_path}")

    if not show:
        plt.close()
    else:
        plt.show()
