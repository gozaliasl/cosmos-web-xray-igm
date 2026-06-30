#!/usr/bin/env python3
"""
Generate showcase images for ALL detected X-ray groups.

For each detected group in both CW-All and CW-HCG catalogs, creates a zoomed
cutout with:
- X-ray emission (contours and filled)
- 2D density map of group member galaxies
- Group center, X-ray peak center, and BCG location
- Aperture and background annulus overlays
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import yaml
from astropy.cosmology import FlatLambdaCDM, LambdaCDM
from astropy.table import Table
from tqdm import tqdm

from src.xray_analysis.data_loader import load_xray_maps
from src.xray_analysis.visualization import plot_group_showcase

# Import load_xray_maps_from_config from main_analysis
sys.path.insert(0, str(Path(__file__).parent))
from main_analysis import load_xray_maps_from_config
from make_showcase_images import (
    load_config, build_cosmology, slugify, load_group_membership,
    find_row_by_group_id
)

logger = logging.getLogger("make_all_showcase_images")


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def make_showcase_for_group(
    catalog_name: str,
    row: Table.Row,
    table: Table,
    xray_maps: Dict,
    cosmology,
    output_dir: Path,
    membership_data_dir: Optional[Path] = None,
    show_member_density: bool = True,
    min_members_for_density: int = 6,
    group_id: Optional[str] = None,
) -> bool:
    """
    Create showcase image for a single group.
    
    Returns True if successful, False otherwise.
    """
    try:
        ra = float(row["RA"])
        dec = float(row["DEC"])
        redshift = float(row["Redshift"])
        is_detected = bool(row.get("Is_Detected", False))
        snr = float(row.get("SNR", np.nan))
        
        # Get X-ray peak coordinates if available
        xray_peak_ra = None
        xray_peak_dec = None
        if "RA_xray_peak" in table.colnames and "Dec_xray_peak" in table.colnames:
            peak_ra_val = row.get("RA_xray_peak")
            peak_dec_val = row.get("Dec_xray_peak")
            if peak_ra_val is not None and peak_dec_val is not None:
                try:
                    peak_ra_float = float(peak_ra_val)
                    peak_dec_float = float(peak_dec_val)
                    if np.isfinite(peak_ra_float) and np.isfinite(peak_dec_float):
                        xray_peak_ra = peak_ra_float
                        xray_peak_dec = peak_dec_float
                except (ValueError, TypeError):
                    pass
        
        # Get Group ID (use provided one or extract from row)
        if group_id is None:
            for col_name in ["Group_ID", "ID", "group_id", "id", "GroupID"]:
                if col_name in table.colnames:
                    group_id_val = row.get(col_name)
                    if group_id_val is not None:
                        group_id = str(group_id_val)
                        break
        
        if group_id is None:
            logger.warning("No Group_ID found for group at RA=%.4f, Dec=%.4f", ra, dec)
            return False
        
        # Get aperture and radii
        aperture_kpc_actual = float(row.get("Aperture_kpc", 300.0))
        r500_kpc = float(row.get("R500_kpc", np.nan)) if "R500_kpc" in table.colnames else np.nan
        r200_kpc = float(row.get("R200_kpc", np.nan)) if "R200_kpc" in table.colnames else np.nan
        
        # Get actual background radii used in analysis (from adaptive background if available)
        bg_inner_kpc_actual = float(row.get("Background_Inner_kpc", np.nan)) if "Background_Inner_kpc" in table.colnames else np.nan
        bg_outer_kpc_actual = float(row.get("Background_Outer_kpc", np.nan)) if "Background_Outer_kpc" in table.colnames else np.nan
        
        # Select appropriate X-ray map
        if 'full' in xray_maps:
            xray_map = xray_maps['full']
            map_type = "full (unmasked)"
        elif 'single' in xray_maps:
            xray_map = xray_maps['single']
            map_type = "single"
        else:
            logger.error("No suitable X-ray map found")
            return False
        
        # Create output filename
        slug = slugify(catalog_name)
        safe_group_id = str(group_id).replace('/', '_').replace('\\', '_')
        output_path = output_dir / f"showcase_{slug}_group_{safe_group_id}.png"
        
        # Load group membership
        member_ra, member_dec, bcg_ra, bcg_dec = None, None, None, None
        if membership_data_dir is not None and group_id is not None:
            member_ra, member_dec, bcg_ra, bcg_dec = load_group_membership(
                catalog_name, group_id, membership_data_dir
            )
        
        # Use actual values from analysis results table (includes adaptive background if applied)
        # Fallback to R500-based calculation if results table values are missing
        if np.isfinite(bg_inner_kpc_actual) and np.isfinite(bg_outer_kpc_actual) and bg_inner_kpc_actual > 0 and bg_outer_kpc_actual > 0:
            # Use actual background radii from results table (reflects adaptive background if used)
            bg_inner_kpc = bg_inner_kpc_actual
            bg_outer_kpc = bg_outer_kpc_actual
        elif is_detected and np.isfinite(r500_kpc) and r500_kpc > 0:
            # Fallback: Use R500-based calculation if results table values are missing
            bg_inner_kpc = r500_kpc * 1.5
            bg_outer_kpc = r500_kpc * 2.0
        else:
            # Default fixed values
            bg_inner_kpc = 500.0
            bg_outer_kpc = 600.0
        
        # Determine source radius
        if is_detected and np.isfinite(r500_kpc) and r500_kpc > 0:
            source_radius_kpc = max(aperture_kpc_actual, r500_kpc)
        else:
            source_radius_kpc = aperture_kpc_actual
        
        # Create showcase plot
        plot_group_showcase(
            xray_map=xray_map,
            ra=ra,
            dec=dec,
            redshift=redshift,
            cosmology=cosmology,
            output_path=str(output_path),
            title=None,
            source_radius_kpc=source_radius_kpc,
            background_inner_kpc=bg_inner_kpc,
            background_outer_kpc=bg_outer_kpc,
            find_xray_peak=(xray_peak_ra is None),  # Only find if not already in catalog
            use_xray_center=False,
            r500_kpc=r500_kpc if np.isfinite(r500_kpc) else None,
            r200_kpc=r200_kpc if np.isfinite(r200_kpc) else None,
            aperture_kpc_actual=aperture_kpc_actual,
            group_id=group_id,
            snr=snr if np.isfinite(snr) else None,
            xray_map_label=map_type,
            member_ra=member_ra,
            member_dec=member_dec,
            bcg_ra=bcg_ra,
            bcg_dec=bcg_dec,
            show_member_density=show_member_density,
            min_members_for_density=min_members_for_density,
            xray_peak_ra=xray_peak_ra,
            xray_peak_dec=xray_peak_dec,
        )
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to create showcase for group: {e}", exc_info=True)
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate showcase images for ALL detected X-ray groups."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--catalog",
        action="append",
        default=None,
        help="Catalog name(s) to process (default: process all configured catalogs).",
    )
    parser.add_argument(
        "--min-snr",
        type=float,
        default=None,
        metavar="S",
        help="Only process groups with SNR >= S (default: process all detected groups).",
    )
    parser.add_argument(
        "--max-groups",
        type=int,
        default=None,
        metavar="N",
        help="Maximum number of groups to process per catalog (for testing).",
    )
    parser.add_argument(
        "--no-density",
        action="store_true",
        help="Disable 2D density map of group members.",
    )
    parser.add_argument(
        "--group-id",
        type=str,
        default=None,
        metavar="ID",
        help="Generate showcase for specific Group_ID only (overrides other selection criteria).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Increase logging verbosity.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config = load_config(config_path)

    cosmology = build_cosmology(config)

    # Load X-ray maps
    xray_maps = load_xray_maps_from_config(config)
    
    if 'full' not in xray_maps and 'single' not in xray_maps:
        logger.error("No 'full' or 'single' X-ray map found.")
        raise ValueError("X-ray map configuration must include 'full' map.")

    output_base = Path(config["output"]["figures_dir"]).resolve()

    catalogs = config.get("data", {}).get("catalogs", [])
    if not catalogs:
        raise ValueError("No catalogs defined in configuration.")

    requested = set(args.catalog) if args.catalog else None

    for entry in catalogs:
        name = entry.get("name")
        if not name:
            continue
        slug = slugify(name)
        if requested and name not in requested and slug not in requested:
            continue
        results_dir = Path(config["output"]["results_dir"]) / slug
        figures_dir = output_base / slug / "showcase_all"
        figures_dir.mkdir(parents=True, exist_ok=True)

        try:
            table = Table.read(results_dir / "xray_catalog.fits")
        except FileNotFoundError as err:
            logger.warning("Skipping %s: %s", name, err)
            continue

        # Handle specific group ID request
        if args.group_id is not None:
            from make_showcase_images import find_row_by_group_id
            row_idx = find_row_by_group_id(table, str(args.group_id).strip())
            if row_idx is None:
                logger.warning("Group_ID '%s' not found in catalog '%s'. Skipping.", args.group_id, name)
                continue
            logger.info("Generating showcase for Group_ID = %s (row index %d)", args.group_id, row_idx)
            detected_indices = [row_idx]
        else:
            # Filter for detected groups
            if "Is_Detected" not in table.colnames:
                logger.warning("No 'Is_Detected' column in %s. Skipping.", name)
                continue
            
            detected = np.asarray(table["Is_Detected"], dtype=bool)
            
            # Apply SNR filter if requested
            if args.min_snr is not None and "SNR" in table.colnames:
                snr_arr = np.asarray(table["SNR"], dtype=float)
                detected = detected & (np.isfinite(snr_arr) & (snr_arr >= args.min_snr))
            
            n_detected = np.sum(detected)
            if n_detected == 0:
                logger.warning("No detected groups found in %s (with SNR >= %.1f).", 
                              name, args.min_snr if args.min_snr is not None else 0)
                continue
            
            logger.info("Processing %d detected groups from %s", n_detected, name)
            
            # Limit number of groups if requested
            detected_indices = np.where(detected)[0]
            if args.max_groups is not None:
                detected_indices = detected_indices[:args.max_groups]
                logger.info("Limiting to first %d groups", len(detected_indices))
        
        # Membership files directory
        membership_data_dir = Path(entry["group_catalog"]).parent if entry.get("group_catalog") else None
        
        # CW-HCG has few members per group: only show positions as small red +, no 2D density map
        min_members_for_density = 999 if name == "CW-HCG" else 6
        
        # Process each detected group
        successful = 0
        failed = 0
        
        for idx in tqdm(detected_indices, desc=f"Processing {name}"):
            row = table[idx]
            # Get group ID for this row
            row_group_id = None
            for col_name in ["Group_ID", "ID", "group_id", "id", "GroupID"]:
                if col_name in table.colnames:
                    group_id_val = row.get(col_name)
                    if group_id_val is not None:
                        row_group_id = str(group_id_val)
                        break
            
            success = make_showcase_for_group(
                catalog_name=name,
                row=row,
                table=table,
                xray_maps=xray_maps,
                cosmology=cosmology,
                output_dir=figures_dir,
                membership_data_dir=membership_data_dir,
                show_member_density=not args.no_density,
                min_members_for_density=min_members_for_density,
                group_id=row_group_id,
            )
            if success:
                successful += 1
            else:
                failed += 1
        
        logger.info("Completed %s: %d successful, %d failed", name, successful, failed)
        logger.info("Showcase images saved to: %s", figures_dir)


if __name__ == "__main__":
    main()
