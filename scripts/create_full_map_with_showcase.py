#!/usr/bin/env python3
"""
Create full X-ray map with showcase inset showing zoomed view of highest SNR detected group.

This script combines the full map visualization with a showcase inset positioned at the group location.
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import ZScaleInterval, ImageNormalize, PercentileInterval, LinearStretch
from astropy.coordinates import SkyCoord
from astropy import units as u
from pathlib import Path
import matplotlib
from matplotlib.patches import Polygon, Rectangle, FancyBboxPatch, ConnectionPatch, Patch, Circle
from matplotlib.lines import Line2D
from matplotlib.transforms import Affine2D
from scipy.ndimage import gaussian_filter
from astropy.table import Table
import yaml
from astropy.cosmology import FlatLambdaCDM
import sys

# Import showcase functionality
sys.path.insert(0, str(Path(__file__).parent))
from make_showcase_images import load_config, build_cosmology, pick_showcase_row, load_group_membership
from main_analysis import load_xray_maps_from_config
from src.xray_analysis.data_loader import load_xray_maps, XrayMap
from astropy.nddata import Cutout2D

matplotlib.use('Agg')  # Non-interactive backend

# Paths
BASE_DIR = Path('/Users/gozalig1/Projects/compact-groups-xray-analysis')
OUTPUT_DIR = BASE_DIR / 'cosmos-web_galaxy-groups-X-ray-properties' / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# X-ray map files
XRAY_MAP = BASE_DIR / 'data' / 'xray-map' / 'cosmos_chaxmm14_520.fits'
XRAY_ERR = BASE_DIR / 'data' / 'xray-map' / 'cosmos_chaxmm14_520_err.fits'


def load_showcase_group_coords(target_group_id="1"):
    """Load specific group coordinates and properties from CW-All catalog (default: Group 1)."""
    config_path = BASE_DIR / 'config.yaml'
    config = load_config(config_path)
    
    # Find CW-All catalog
    catalogs = config.get("data", {}).get("catalogs", [])
    cw_all_entry = None
    for entry in catalogs:
        if entry.get("name", "").strip().lower() in ("cw-all", "cw_all"):
            cw_all_entry = entry
            break
    
    if cw_all_entry is None:
        print("Warning: CW-All catalog not found in config")
        return None, None, None, None, None, None, None, None, None, None, None, None, None, None
    
    # Load results table
    slug = "cw_all"
    results_dir = Path(config["output"]["results_dir"]) / slug
    results_table_path = results_dir / "xray_catalog.fits"
    
    if not results_table_path.exists():
        print(f"Warning: Results table not found: {results_table_path}")
        return None, None, None, None, None, None, None, None, None, None, None, None, None, None
    
    table = Table.read(results_table_path)
    
    # Find Group 1
    from make_showcase_images import find_row_by_group_id
    row_idx = find_row_by_group_id(table, str(target_group_id))
    if row_idx is None:
        print(f"Warning: Group {target_group_id} not found")
        return None, None, None, None, None, None, None, None, None, None, None, None, None, None
    
    row = table[row_idx]
    ra = float(row["RA"])
    dec = float(row["DEC"])
    redshift = float(row["Redshift"])
    snr = float(row.get("SNR", np.nan))
    r500_kpc = float(row.get("R500_kpc", np.nan)) if "R500_kpc" in table.colnames else np.nan
    r200_kpc = float(row.get("R200_kpc", np.nan)) if "R200_kpc" in table.colnames else np.nan
    aperture_kpc_actual = float(row.get("Aperture_kpc", 300.0))
    bg_inner_kpc = float(row.get("Background_Inner_kpc", np.nan)) if "Background_Inner_kpc" in table.colnames else np.nan
    bg_outer_kpc = float(row.get("Background_Outer_kpc", np.nan)) if "Background_Outer_kpc" in table.colnames else np.nan
    
    # Fallback for background radii
    if not (np.isfinite(bg_inner_kpc) and np.isfinite(bg_outer_kpc) and bg_inner_kpc > 0):
        if np.isfinite(r500_kpc) and r500_kpc > 0:
            bg_inner_kpc = r500_kpc * 1.5
            bg_outer_kpc = r500_kpc * 2.0
        else:
            bg_inner_kpc = 500.0
            bg_outer_kpc = 600.0
    
    # Get group ID
    group_id = str(target_group_id)
    
    # Load membership data
    membership_data_dir = Path(cw_all_entry.get("group_catalog", "")).parent if cw_all_entry.get("group_catalog") else None
    member_ra, member_dec, bcg_ra, bcg_dec = None, None, None, None
    if membership_data_dir and membership_data_dir.exists():
        member_ra, member_dec, bcg_ra, bcg_dec = load_group_membership(
            cw_all_entry.get("name", ""), group_id, membership_data_dir
        )
    
    print(f"Selected showcase group: ID={group_id}, RA={ra:.4f}°, Dec={dec:.4f}°, z={redshift:.3f}, SNR={snr:.2f}, R500={r500_kpc:.1f} kpc")
    return ra, dec, group_id, snr, r500_kpc, redshift, r200_kpc, aperture_kpc_actual, bg_inner_kpc, bg_outer_kpc, member_ra, member_dec, bcg_ra, bcg_dec


def plot_showcase_inset(ax_inset, xray_map, ra, dec, redshift, cosmology, r500_kpc, r200_kpc, 
                        aperture_kpc_actual, bg_inner_kpc, bg_outer_kpc, group_id, snr,
                        member_ra=None, member_dec=None, bcg_ra=None, bcg_dec=None):
    """Plot showcase group directly into inset axes with all original information."""
    from src.xray_analysis.peak_finding import find_xray_centroid
    
    pixel_scale = xray_map.pixel_scale  # arcsec per pixel
    
    # Calculate cutout size based on background outer radius to ensure it's visible
    # Make cutout large enough to show the outer background circle with margin
    da_kpc = cosmology.angular_diameter_distance(redshift).to(u.kpc).value
    bg_outer_arcsec = (bg_outer_kpc / da_kpc) * 206265
    if np.isfinite(r500_kpc) and r500_kpc > 0:
        r500_arcsec = (r500_kpc / da_kpc) * 206265
        # Use much larger factor to ensure outer background circle is fully visible
        # Need at least 2.2x the outer radius to show the full circle with margin
        cutout_size_arcsec = max(r500_arcsec * 2.5, bg_outer_arcsec * 2.2)
    else:
        cutout_size_arcsec = bg_outer_arcsec * 2.2
    
    half_size_pix = max(10, int(np.ceil(cutout_size_arcsec / pixel_scale / 2)))
    cutout_size = 2 * half_size_pix + 1
    
    # Convert to pixel coordinates
    position = xray_map.world_to_pixel(np.array([ra]), np.array([dec]))
    position = (position[0][0], position[1][0])
    
    # Create cutout
    cutout = Cutout2D(xray_map.data, position=position, size=(cutout_size, cutout_size),
                      wcs=xray_map.wcs, mode='partial', fill_value=np.nan)
    data = cutout.data
    
    # Find X-ray peak
    xray_peak_ra, xray_peak_dec, xray_peak_snr = None, None, None
    try:
        peak_x, peak_y, peak_snr = find_xray_centroid(
            data=xray_map.data,
            initial_x=position[0],
            initial_y=position[1],
            aperture_radius_pix=aperture_kpc_actual / cosmology.angular_diameter_distance(redshift).to(u.kpc).value * 206265 / pixel_scale,
            smoothing_sigma=1.2,
            min_snr=2.0,
            error_map=xray_map.error if hasattr(xray_map, 'error') else None
        )
        if peak_x is not None:
            peak_coords = xray_map.wcs.pixel_to_world(peak_x, peak_y)
            xray_peak_ra = float(peak_coords.ra.deg)
            xray_peak_dec = float(peak_coords.dec.deg)
            xray_peak_snr = peak_snr
    except:
        pass
    
    # Prepare data for contours
    data_for_contours = np.array(data, copy=True)
    finite_mask = np.isfinite(data_for_contours)
    if not np.any(finite_mask):
        return
    median_val = np.nanmedian(data_for_contours[finite_mask])
    data_for_contours[~finite_mask] = median_val
    
    # Smooth for contours
    contour_smoothing_sigma = 1.2
    contour_smoothing_for_extended = 2.0
    sigma_extended = contour_smoothing_sigma + contour_smoothing_for_extended
    smoothed_for_contours = gaussian_filter(data_for_contours, sigma=sigma_extended)
    
    # Plot background - make transparent
    ny, nx = data.shape
    ax_inset.set_facecolor('none')  # Transparent background
    ax_inset.set_aspect('equal')
    
    # Plot X-ray contours
    smin = np.nanmin(smoothed_for_contours)
    smax = np.nanmax(smoothed_for_contours)
    if np.isfinite(smin) and np.isfinite(smax) and smax > smin:
        contour_percentiles = [65, 75, 85, 90, 95, 98]
        contour_levels = np.nanpercentile(smoothed_for_contours, contour_percentiles)
        contour_levels = np.unique(contour_levels[np.isfinite(contour_levels)])
        eps = max(1e-10 * (smax - smin), 1e-15)
        contour_levels = contour_levels[contour_levels > smin + eps]
        if contour_levels.size > 0:
            x_pix = np.arange(nx)
            y_pix = np.arange(ny)
            ax_inset.contourf(x_pix, y_pix, smoothed_for_contours, levels=contour_levels, 
                              cmap='viridis', alpha=0.68, zorder=7, extend='min')
            ax_inset.contour(x_pix, y_pix, smoothed_for_contours, levels=contour_levels, 
                            colors='white', linewidths=0.6, alpha=0.85, zorder=8)
    
    # Calculate positions in cutout
    center_x, center_y = (nx - 1) / 2.0, (ny - 1) / 2.0
    
    # Convert physical radii to arcsec and pixels
    def kpc_to_arcsec(kpc_val):
        da_kpc = cosmology.angular_diameter_distance(redshift).to(u.kpc).value
        return (kpc_val / da_kpc) * 206265
    
    # Add circles
    def add_circle(radius_arcsec, color, linestyle='-', linewidth=2.0):
        radius_pix = radius_arcsec / pixel_scale
        circle = Circle((center_x, center_y), radius_pix, edgecolor=color, facecolor='none',
                       linewidth=linewidth, linestyle=linestyle, alpha=1.0, zorder=15)
        ax_inset.add_patch(circle)
        return circle
    
    # Aperture circle
    aperture_arcsec = kpc_to_arcsec(aperture_kpc_actual)
    add_circle(aperture_arcsec, 'lime', linewidth=2.0)
    
    # R500 and R200 circles
    if np.isfinite(r500_kpc) and r500_kpc > 0:
        r500_arcsec = kpc_to_arcsec(r500_kpc)
        add_circle(r500_arcsec, 'magenta', linewidth=2.0)
    if np.isfinite(r200_kpc) and r200_kpc > 0:
        r200_arcsec = kpc_to_arcsec(r200_kpc)
        add_circle(r200_arcsec, 'purple', linewidth=2.0)
    
    # Background annulus circles
    bg_inner_arcsec = kpc_to_arcsec(bg_inner_kpc)
    bg_outer_arcsec = kpc_to_arcsec(bg_outer_kpc)
    add_circle(bg_inner_arcsec, 'blue', linestyle='--', linewidth=2.0)
    # Make outer circle more visible with thicker linewidth and ensure it's visible
    bg_outer_radius_pix = bg_outer_arcsec / pixel_scale
    bg_outer_circle = Circle((center_x, center_y), bg_outer_radius_pix, 
                            edgecolor='orange', facecolor='none', linewidth=3.0, 
                            linestyle='--', alpha=1.0, zorder=16)  # Higher zorder and thicker line
    ax_inset.add_patch(bg_outer_circle)
    
    # Calculate margin to ensure outer circle is visible
    outer_circle_max_extent = bg_outer_radius_pix
    margin = max(10, int(0.15 * outer_circle_max_extent))  # 15% margin for visibility
    
    # Mark catalog center
    ax_inset.plot(center_x, center_y, 'rx', markersize=10, markeredgewidth=2,
                 markeredgecolor='black', markerfacecolor='red', zorder=10)
    
    # Mark X-ray peak if found
    if xray_peak_ra is not None:
        peak_pos = cutout.wcs.world_to_pixel(SkyCoord(ra=xray_peak_ra*u.deg, dec=xray_peak_dec*u.deg))
        peak_offset_x = peak_pos[0] - position[0]
        peak_offset_y = peak_pos[1] - position[1]
        peak_x_in_cutout = center_x + peak_offset_x
        peak_y_in_cutout = center_y + peak_offset_y
        if 0 <= peak_x_in_cutout < nx and 0 <= peak_y_in_cutout < ny:
            ax_inset.plot(peak_x_in_cutout, peak_y_in_cutout, 'g+', markersize=12,
                         markeredgewidth=2.5, markeredgecolor='black', zorder=11)
    
    # Plot member galaxy density map instead of individual points
    if member_ra is not None and len(member_ra) > 0:
        from scipy.stats import gaussian_kde
        member_pos = xray_map.world_to_pixel(member_ra, member_dec)
        member_x_in_cutout = []
        member_y_in_cutout = []
        for mx, my in zip(member_pos[0], member_pos[1]):
            offset_x = mx - position[0]
            offset_y = my - position[1]
            mx_cutout = center_x + offset_x
            my_cutout = center_y + offset_y
            if 0 <= mx_cutout < nx and 0 <= my_cutout < ny:
                member_x_in_cutout.append(mx_cutout)
                member_y_in_cutout.append(my_cutout)
        
        if len(member_x_in_cutout) >= 3:  # Need at least 3 points for KDE
            try:
                # Filter valid values
                member_x_arr = np.array(member_x_in_cutout)
                member_y_arr = np.array(member_y_in_cutout)
                valid = np.isfinite(member_x_arr) & np.isfinite(member_y_arr)
                if np.sum(valid) >= 3:
                    x_valid = member_x_arr[valid]
                    y_valid = member_y_arr[valid]
                    
                    # Create grid for density evaluation
                    x_grid = np.linspace(0, nx - 1, nx)
                    y_grid = np.linspace(0, ny - 1, ny)
                    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
                    
                    # Calculate KDE
                    try:
                        kde = gaussian_kde(np.vstack([x_valid, y_valid]))
                        positions = np.vstack([X_grid.ravel(), Y_grid.ravel()])
                        density = kde(positions).reshape(X_grid.shape)
                    except:
                        # Fallback: use 2D histogram with smoothing
                        density, x_edges, y_edges = np.histogram2d(
                            x_valid, y_valid, bins=[nx//4, ny//4],
                            range=[[0, nx-1], [0, ny-1]]
                        )
                        density = density / np.max(density) if np.max(density) > 0 else density
                        density = gaussian_filter(density, sigma=2.0)
                        # Interpolate to full grid
                        from scipy.interpolate import RectBivariateSpline
                        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
                        y_centers = (y_edges[:-1] + y_edges[1:]) / 2
                        interp = RectBivariateSpline(y_centers, x_centers, density, kx=1, ky=1)
                        density = interp(y_grid, x_grid)
                    
                    # Normalize density
                    if np.max(density) > 0:
                        density_norm = density / np.max(density)
                    else:
                        density_norm = density
                    
                    # Define contour levels
                    density_flat = density_norm.flatten()
                    density_flat = density_flat[density_flat > 0]
                    if len(density_flat) > 0:
                        percentiles = np.percentile(density_flat, [20, 40, 60, 80])
                        levels = percentiles[percentiles > 0.05]
                        if len(levels) == 0:
                            levels = np.linspace(0.1, 0.9, 4)
                    else:
                        levels = np.linspace(0.1, 0.9, 4)
                    
                    # Plot density contours with solid cyan
                    if len(levels) > 0:
                        ax_inset.contour(X_grid, Y_grid, density_norm, levels=levels,
                                        colors='cyan', linewidths=1.5, alpha=0.7, zorder=12, linestyles='-')
            except Exception as e:
                # Fallback to individual points if density map fails
                if member_x_in_cutout:
                    ax_inset.plot(member_x_in_cutout, member_y_in_cutout, '+', color='red',
                                 markersize=6, markeredgewidth=1.5, markeredgecolor='darkred', zorder=12)
    
    # Plot BCG
    if bcg_ra is not None and bcg_dec is not None:
        bcg_pos = xray_map.world_to_pixel(np.array([bcg_ra]), np.array([bcg_dec]))
        bcg_offset_x = bcg_pos[0][0] - position[0]
        bcg_offset_y = bcg_pos[1][0] - position[1]
        bcg_x_in_cutout = center_x + bcg_offset_x
        bcg_y_in_cutout = center_y + bcg_offset_y
        if 0 <= bcg_x_in_cutout < nx and 0 <= bcg_y_in_cutout < ny:
            ax_inset.plot(bcg_x_in_cutout, bcg_y_in_cutout, '*', color='gold',
                         markersize=10, markeredgewidth=2, markeredgecolor='black', zorder=13)
    
    # Set labels (smaller font for inset)
    ax_inset.set_xlabel('Pixel x', fontsize=9, fontweight='bold')
    ax_inset.set_ylabel('Pixel y', fontsize=9, fontweight='bold')
    ax_inset.tick_params(labelsize=8)
    
    # Add legend (compact version) with circle information
    from matplotlib.patches import Patch, Rectangle
    from matplotlib.lines import Line2D
    handles = []
    labels = []
    empty_patch = Rectangle((0, 0), 0, 0, fill=False, edgecolor='none', visible=False)
    
    if group_id:
        handles.append(empty_patch)
        labels.append(f'ID={group_id}')
    handles.append(empty_patch)
    labels.append(f'RA={ra:.4f}°')
    handles.append(empty_patch)
    labels.append(f'Dec={dec:.4f}°')
    handles.append(empty_patch)
    labels.append(f'$z$={redshift:.3f}')
    if np.isfinite(snr):
        handles.append(empty_patch)
        labels.append(f'SNR={snr:.2f}')
    
    # Add circle entries to legend
    # Aperture circle
    handles.append(Circle((0, 0), 1, edgecolor='lime', facecolor='none', linewidth=2.0))
    labels.append(f'Aperture ({int(aperture_kpc_actual)} kpc)')
    
    # R500 circle
    if np.isfinite(r500_kpc) and r500_kpc > 0:
        handles.append(Circle((0, 0), 1, edgecolor='magenta', facecolor='none', linewidth=2.0))
        labels.append(f'$R_{{500}}$ ({int(r500_kpc)} kpc)')
    
    # R200 circle
    if np.isfinite(r200_kpc) and r200_kpc > 0:
        handles.append(Circle((0, 0), 1, edgecolor='purple', facecolor='none', linewidth=2.0))
        labels.append(f'$R_{{200}}$ ({int(r200_kpc)} kpc)')
    
    # Background annulus circles
    handles.append(Circle((0, 0), 1, edgecolor='blue', facecolor='none', linewidth=2.0, linestyle='--'))
    labels.append(f'Bg inner ({int(bg_inner_kpc)} kpc)')
    handles.append(Circle((0, 0), 1, edgecolor='orange', facecolor='none', linewidth=2.0, linestyle='--'))
    labels.append(f'Bg outer ({int(bg_outer_kpc)} kpc)')
    
    # Add galaxy overdensity map (solid cyan)
    handles.append(Line2D([0], [0], color='cyan', linestyle='-', linewidth=1.5))
    labels.append('Galaxy overdensity')
    
    # Add catalog center (red X)
    handles.append(Line2D([0], [0], marker='x', color='red', markeredgecolor='black', 
                         markeredgewidth=2, markersize=8, linestyle='None'))
    labels.append('Catalog center')
    
    # Add BCG (gold star) if available
    if bcg_ra is not None and bcg_dec is not None:
        handles.append(Line2D([0], [0], marker='*', color='gold', markeredgecolor='black',
                             markeredgewidth=2, markersize=10, linestyle='None'))
        labels.append('BCG')
    
    if handles:
        # Move legend to top left, well outside the plot area
        legend = ax_inset.legend(handles, labels, loc='upper left', bbox_to_anchor=(-2.0, 1.0),
                                fontsize=7, frameon=False, handletextpad=0.3, labelspacing=0.4)
        for text in legend.get_texts():
            text.set_fontweight('bold')
    
    # Remove gridlines
    ax_inset.grid(False)
    # Set axis limits to show the full cutout including outer background circle
    # Zoom in to show outer circle with proper margin
    ax_inset.set_xlim(center_x - outer_circle_max_extent - margin, 
                     center_x + outer_circle_max_extent + margin)
    ax_inset.set_ylim(center_y - outer_circle_max_extent - margin, 
                     center_y + outer_circle_max_extent + margin)


def create_full_map_with_showcase_inset():
    """Create full X-ray map with showcase inset."""
    print("Loading X-ray maps...")
    
    # Load XrayMap object
    xray_map = load_xray_maps(str(XRAY_MAP), str(XRAY_ERR), verbose=False)
    data = xray_map.data
    wcs = xray_map.wcs
    
    # Get showcase group coordinates (Group 1) with all properties
    result = load_showcase_group_coords(target_group_id="1")
    if result[0] is None:
        print("Warning: Could not load showcase group")
        return
    showcase_ra, showcase_dec, group_id, showcase_snr, r500_kpc, redshift, r200_kpc, aperture_kpc_actual, bg_inner_kpc, bg_outer_kpc, member_ra, member_dec, bcg_ra, bcg_dec = result
    
    # Build cosmology for distance calculations
    config_path = BASE_DIR / 'config.yaml'
    config = load_config(config_path)
    cosmology = build_cosmology(config)
    
    # Create figure
    fig = plt.figure(figsize=(14, 10))
    ax = plt.subplot(projection=wcs)
    
    # Show more of the map - reduce crop factor to show larger area
    crop_factor = 0.95  # Show 95% of the map (was 0.85)
    center_x, center_y = data.shape[1] / 2, data.shape[0] / 2
    width = data.shape[1] * crop_factor
    height = data.shape[0] * crop_factor
    ax.set_xlim(center_x - width/2, center_x + width/2)
    ax.set_ylim(center_y - height/2, center_y + height/2)
    
    # Prepare display data
    masked_data = np.ma.masked_invalid(data)
    display_data = np.where(masked_data.mask, np.nan, masked_data.data)
    
    # Use symmetric scaling centered at zero
    valid_data = display_data[~np.isnan(display_data)]
    if len(valid_data) > 0:
        p99 = np.percentile(np.abs(valid_data), 99)
        vmin, vmax = -p99, p99
    else:
        vmin, vmax = -1e-4, 1e-4
    
    norm = ImageNormalize(display_data, vmin=vmin, vmax=vmax)
    
    # Plot the map
    im = ax.imshow(display_data, origin='lower', cmap='viridis', norm=norm, aspect='equal')
    
    # Create X-ray contours
    data_for_contours = np.array(display_data, copy=True)
    finite_mask = np.isfinite(data_for_contours)
    if np.any(finite_mask):
        median_val = np.nanmedian(data_for_contours[finite_mask])
        data_for_contours[~finite_mask] = median_val
        
        contour_smoothing_sigma = 1.2
        contour_smoothing_for_extended = 2.0
        sigma_extended = contour_smoothing_sigma + contour_smoothing_for_extended
        smoothed_for_contours = gaussian_filter(data_for_contours, sigma=sigma_extended)
        
        positive_data = smoothed_for_contours[smoothed_for_contours > 0]
        if len(positive_data) > 0:
            contour_percentiles = np.linspace(60, 95, 4)
            contour_levels = np.nanpercentile(positive_data, contour_percentiles)
            contour_levels = np.unique(contour_levels[np.isfinite(contour_levels)])
            eps = max(1e-10 * (np.nanmax(smoothed_for_contours) - np.nanmin(smoothed_for_contours)), 1e-15)
            contour_levels = contour_levels[contour_levels > np.nanmin(smoothed_for_contours) + eps]
            
            if len(contour_levels) > 0:
                contours = ax.contour(smoothed_for_contours, levels=contour_levels, colors='white',
                                     linewidths=0.6, alpha=0.85, linestyles='solid', origin='lower', zorder=5)
    
    # Set coordinate labels with degree symbol
    ax.set_xlabel('Right Ascension (deg)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Declination (deg)', fontsize=12, fontweight='bold')
    
    # Format coordinate ticks - show in degrees (decimal format) with more spacing
    ax.coords[0].set_format_unit('deg')
    ax.coords[1].set_format_unit('deg')
    # Increase tick spacing to avoid crowding (from 0.1 to 0.15 degrees)
    # Make ticks visible with black color and proper size
    ax.coords[0].set_ticks(spacing=0.15*u.deg, color='black', size=6, width=1.5)
    ax.coords[1].set_ticks(spacing=0.15*u.deg, color='black', size=6, width=1.5)
    ax.coords[0].set_ticklabel(size=10, color='black')
    ax.coords[1].set_ticklabel(size=10, color='black')
    # Add more padding between tick labels
    ax.coords[0].set_ticklabel(pad=8)
    ax.coords[1].set_ticklabel(pad=8)
    # Enable tick marks on both axes (disable minor ticks, major ticks are shown by default)
    ax.coords[0].display_minor_ticks(False)
    ax.coords[1].display_minor_ticks(False)
    
    # Add COSMOS-Web footprint overlay
    cosmos_center = SkyCoord(ra="10h00m27.92s", dec="+02d12m03.5s", frame='fk5')
    cosmos_width = 0.54**0.5 * u.deg
    cosmos_height = 0.54**0.5 * u.deg
    angle = 110
    
    center_pix = wcs.world_to_pixel(cosmos_center)
    half_width_deg = cosmos_width.to(u.deg).value / 2
    half_height_deg = cosmos_height.to(u.deg).value / 2
    
    corners_ra = np.array([cosmos_center.ra.deg - half_width_deg, cosmos_center.ra.deg + half_width_deg,
                           cosmos_center.ra.deg + half_width_deg, cosmos_center.ra.deg - half_width_deg])
    corners_dec = np.array([cosmos_center.dec.deg - half_height_deg, cosmos_center.dec.deg - half_height_deg,
                            cosmos_center.dec.deg + half_height_deg, cosmos_center.dec.deg + half_height_deg])
    
    corners_pix = wcs.world_to_pixel(SkyCoord(ra=corners_ra*u.deg, dec=corners_dec*u.deg, frame='fk5'))
    rotation_transform = Affine2D().rotate_deg_around(center_pix[0], center_pix[1], angle)
    rotated_corners = rotation_transform.transform(np.column_stack([corners_pix[0], corners_pix[1]]))
    
    footprint_color = 'black'
    polygon = Polygon(rotated_corners, edgecolor=footprint_color, facecolor='none', 
                     linestyle='--', lw=2.5, alpha=0.9, zorder=10)
    ax.add_patch(polygon)
    
    # Add COSMOS-Web label at the center of the footprint with transparent background
    center_pix = wcs.world_to_pixel(cosmos_center)
    ax.text(center_pix[0], center_pix[1], 'COSMOS-Web', color=footprint_color, fontsize=12, 
            ha='center', va='center', weight='bold', zorder=11)
    
    # Add showcase group location marker and zoom box
    if showcase_ra is not None and showcase_dec is not None:
        # Convert showcase group position to pixel coordinates
        showcase_pix = wcs.world_to_pixel(SkyCoord(ra=showcase_ra*u.deg, dec=showcase_dec*u.deg, frame='fk5'))
        showcase_x, showcase_y = showcase_pix[0], showcase_pix[1]
        
        # Draw R500 circle if available
        if np.isfinite(r500_kpc) and r500_kpc > 0 and np.isfinite(redshift) and redshift > 0:
            # Convert R500 from kpc to arcsec
            da_kpc = cosmology.angular_diameter_distance(redshift).to(u.kpc).value
            r500_arcsec = (r500_kpc / da_kpc) * 206265  # Convert to arcsec
            pixel_scale = 4.0  # arcsec per pixel (approximate)
            r500_pix = r500_arcsec / pixel_scale
            
            # Draw R500 circle
            r500_circle = Circle((showcase_x, showcase_y), r500_pix,
                                edgecolor='magenta', facecolor='none', linewidth=1.5,
                                alpha=0.9, zorder=11, linestyle='-')
            ax.add_patch(r500_circle)
            
            # Use R500-based zoom box size (show ~2-3x R500)
            zoom_size_arcsec = r500_arcsec * 2.5
        else:
            # Fallback: Estimate showcase cutout size
            zoom_size_arcsec = 150  # Approximate size of showcase cutout
        
        pixel_scale = 4.0  # arcsec per pixel (approximate)
        zoom_size_pix = zoom_size_arcsec / pixel_scale
        
        # Draw rectangle showing zoom area (magenta color, narrow line)
        zoom_box = Rectangle((showcase_x - zoom_size_pix/2, showcase_y - zoom_size_pix/2),
                             zoom_size_pix, zoom_size_pix,
                             edgecolor='magenta', facecolor='none', linewidth=1.5,
                             linestyle='--', alpha=0.8, zorder=9)
        ax.add_patch(zoom_box)
        
        # Create inset axes for showcase plot (smaller size)
        # Position inset in upper right corner, shifted right by 12% + 20% more = 32% total, then 2% left, and up by 2%
        # Original: [0.55, 0.65, 0.28, 0.28]
        # Shift right: 0.58 + 0.20*0.28 = 0.58 + 0.056 = 0.636 ≈ 0.64
        # Shift left by 2%: 0.64 - 0.02*0.28 = 0.64 - 0.0056 ≈ 0.634 ≈ 0.63
        # Shift up: 0.65 + 0.02*0.28 = 0.6556 ≈ 0.66
        ax_inset = fig.add_axes([0.63, 0.66, 0.28, 0.28])  # [left, bottom, width, height]
        
        # Plot showcase directly into inset axes with all info
        plot_showcase_inset(ax_inset, xray_map, showcase_ra, showcase_dec, redshift, 
                           cosmology, r500_kpc, r200_kpc, aperture_kpc_actual, 
                           bg_inner_kpc, bg_outer_kpc, group_id, showcase_snr,
                           member_ra, member_dec, bcg_ra, bcg_dec)
        
        # Add border around inset (magenta color to match zoom box, narrow line)
        border = FancyBboxPatch((0, 0), 1, 1, transform=ax_inset.transAxes,
                               fill=False, edgecolor='magenta', linewidth=1.5, linestyle='--')
        ax_inset.add_patch(border)
        
        # Draw connecting lines from magenta zoom box to left-bottom corner of inset
        # Right and left corners of zoom box (in main axes data coordinates)
        right_x = showcase_x + zoom_size_pix/2
        left_x = showcase_x - zoom_size_pix/2
        bottom_right_y = showcase_y - zoom_size_pix/2
        top_right_y = showcase_y + zoom_size_pix/2
        
        # Left-bottom corner of inset (in inset axes coordinates: 0, 0)
        inset_left_bottom = (0, 0)
        
        # Bottom-right corner to inset left-bottom
        conn1 = ConnectionPatch(
            xyA=(right_x, bottom_right_y), xyB=inset_left_bottom,
            coordsA='data', coordsB='axes fraction',
            axesA=ax, axesB=ax_inset,
            color='magenta', linewidth=1.2, linestyle='--', alpha=0.7, zorder=8
        )
        ax.add_patch(conn1)
        
        # Top-right corner to inset left-bottom
        conn2 = ConnectionPatch(
            xyA=(right_x, top_right_y), xyB=inset_left_bottom,
            coordsA='data', coordsB='axes fraction',
            axesA=ax, axesB=ax_inset,
            color='magenta', linewidth=1.2, linestyle='--', alpha=0.7, zorder=8
        )
        ax.add_patch(conn2)

        # Top-left corner to inset left-bottom
        top_left_y = top_right_y
        conn3 = ConnectionPatch(
            xyA=(left_x, top_left_y), xyB=inset_left_bottom,
            coordsA='data', coordsB='axes fraction',
            axesA=ax, axesB=ax_inset,
            color='magenta', linewidth=1.2, linestyle='--', alpha=0.7, zorder=8
        )
        ax.add_patch(conn3)
    
    # Add colorbar with bold, larger labels
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Count Rate (counts s$^{-1}$ pixel$^{-1}$)', fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=12)
    
    # Add legend with bold, larger text
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='black', linestyle='--', linewidth=2.5, label='COSMOS-Web footprint'),
    ]
    if showcase_ra is not None:
        # Show showcase zoom area with catalog and group ID (magenta color, narrow line)
        legend_elements.append(
            Line2D([0], [0], color='magenta', linestyle='--', linewidth=1.5, 
                  label=f'Showcase zoom area (CW-All ID={group_id}.0)')
        )
        # Add X-ray contours to legend (white, lw=0.6 to match full map)
        legend_elements.append(
            Line2D([0], [0], color='white', linestyle='-', linewidth=0.6, 
                  label='X-ray contours')
        )
    # Move main map legend to left bottom with smaller font size, shifted 10% to the left from previous position
    legend = ax.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(0.05, 0.0),
                      fontsize=10, frameon=False)
    # Make legend text bold
    for text in legend.get_texts():
        text.set_fontweight('bold')
    
    plt.subplots_adjust(left=0.08, right=0.95, bottom=0.08, top=0.95)
    
    # Save PNG version
    output_path_png = OUTPUT_DIR / 'xray_map_full_with_showcase.png'
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"Saved: {output_path_png}")
    
    # Save PDF version
    output_path_pdf = OUTPUT_DIR / 'xray_map_full_with_showcase.pdf'
    plt.savefig(output_path_pdf, bbox_inches='tight', pad_inches=0.1)
    print(f"Saved: {output_path_pdf}")
    
    plt.close()


if __name__ == '__main__':
    create_full_map_with_showcase_inset()
