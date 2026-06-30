#!/usr/bin/env python3
"""
Generate zoomed X-ray showcase images for selected COSMOS-Web groups.

For each requested catalog (default: CW-All and CW-HCG), the script selects
the highest-SNR DETECTED group (with X-ray emission) and produces a zoomed cutout 
with X-ray contours, the 300 kpc source aperture, and the 500/600 kpc background 
annulus overlaid. Uses the full (unmasked) X-ray map for detected groups.
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

from src.xray_analysis.data_loader import load_xray_maps
from src.xray_analysis.visualization import plot_group_showcase

# Import load_xray_maps_from_config from main_analysis
sys.path.insert(0, str(Path(__file__).parent))
from main_analysis import load_xray_maps_from_config

logger = logging.getLogger("make_showcase_images")


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r") as fh:
        return yaml.safe_load(fh)


def build_cosmology(config: Dict[str, Any]) -> FlatLambdaCDM:
    cosmo_cfg = config.get("cosmology", {})
    H0 = cosmo_cfg.get("H0", 70.0)
    Om0 = cosmo_cfg.get("Om0", 0.3)
    Ode0 = cosmo_cfg.get("Ode0")
    if Ode0 is not None:
        total = Om0 + Ode0
        if abs(total - 1.0) < 1e-3:
            return FlatLambdaCDM(H0=H0, Om0=Om0)
        return LambdaCDM(H0=H0, Om0=Om0, Ode0=Ode0)
    return FlatLambdaCDM(H0=H0, Om0=Om0)


def slugify(value: str) -> str:
    import re

    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "catalog"


def _is_clean_ra_dec(ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
    """
    Return boolean mask True where RA and Dec are valid (not sentinel values).
    Excludes values like -999, 999, 99, -99, 999.99, -999.99, etc.
    """
    ra = np.asarray(ra, dtype=float)
    dec = np.asarray(dec, dtype=float)
    # Sentinel values commonly used for missing data
    bad_vals = (-999.0, -999.99, -99.0, -99.99, 99.0, 99.99, 999.0, 999.99,
                 9999.0, -9999.0, 0.0)  # 0,0 can be placeholder
    clean = np.isfinite(ra) & np.isfinite(dec)
    for bad in bad_vals:
        clean &= (np.abs(ra - bad) > 0.01) & (np.abs(dec - bad) > 0.01)
    # RA typically 0–360, Dec -90–90 for valid sky
    clean &= (ra >= -1) & (ra <= 361) & (dec >= -91) & (dec <= 91)
    return clean


def load_group_membership(
    catalog_name: str,
    group_id: Optional[str],
    data_dir: Path,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float], Optional[float]]:
    """
    Load group membership for a catalog and return member and BCG positions.

    Returns (member_ra, member_dec, bcg_ra, bcg_dec). RA/Dec in degrees.
    BCG is the galaxy with the highest stellar mass (LP_mass_med_PDF); if no mass
    column, no BCG is identified (bcg_ra, bcg_dec = None, None).
    Excludes galaxies with bad RA/Dec (sentinel values like -999, 999, etc.)
    and requires LP_warn_fl == 0 when that column exists.
    """
    if not group_id:
        return None, None, None, None

    data_dir = Path(data_dir)
    member_ra = member_dec = None
    bcg_ra = bcg_dec = None

    # Normalize group_id for matching (catalog may store as int or string)
    try:
        group_id_int = int(group_id)
    except (ValueError, TypeError):
        group_id_int = None

    def match_gid(row_gid) -> bool:
        if row_gid is None:
            return False
        if group_id_int is not None:
            try:
                return int(row_gid) == group_id_int
            except (ValueError, TypeError):
                pass
        return str(row_gid).strip() == str(group_id).strip()

    # CW-All: cosmos_web_groups_membership.fits (or assoc05.csv)
    if catalog_name.strip().lower() in ("cw-all", "cw_all", "cosmos web all"):
        path_fits = data_dir / "cosmos_web_groups_membership.fits"
        if path_fits.exists():
            try:
                mem = Table.read(path_fits)
                if "Group_ID" not in mem.colnames:
                    return None, None, None, None
                mask = np.array([match_gid(g) for g in mem["Group_ID"]])
                if not np.any(mask):
                    return None, None, None, None
                sub = mem[mask]
                # Require LP_warn_fl == 0 when present (clean galaxies only)
                if "LP_warn_fl" in sub.colnames:
                    try:
                        warn = np.asarray(sub["LP_warn_fl"], dtype=float)
                        sub = sub[np.isfinite(warn) & (warn == 0)]
                    except Exception:
                        pass
                if len(sub) == 0:
                    return None, None, None, None
                # For CW-All, prioritize RA_MODEL/DEC_MODEL (individual galaxy positions)
                # over RA/DEC (which may be group center coordinates)
                if "RA_MODEL" in sub.colnames and "DEC_MODEL" in sub.colnames:
                    ra_col = "RA_MODEL"
                    dec_col = "DEC_MODEL"
                elif "RA" in sub.colnames and "DEC" in sub.colnames:
                    ra_col = "RA"
                    dec_col = "DEC"
                else:
                    return None, None, None, None
                member_ra = np.asarray(sub[ra_col], dtype=float)
                member_dec = np.asarray(sub[dec_col], dtype=float)
                # Exclude galaxies with bad/sentinel RA/Dec
                clean = _is_clean_ra_dec(member_ra, member_dec)
                if not np.any(clean):
                    return None, None, None, None
                member_ra = member_ra[clean]
                member_dec = member_dec[clean]
                sub = sub[clean]
                # BCG = highest stellar mass
                if "LP_mass_med_PDF" in sub.colnames:
                    mass = np.asarray(sub["LP_mass_med_PDF"], dtype=float)
                    valid = np.isfinite(mass)
                    if np.any(valid):
                        idx_bcg = np.nanargmax(np.where(valid, mass, -np.inf))
                        bcg_ra = float(member_ra[idx_bcg])
                        bcg_dec = float(member_dec[idx_bcg])
            except Exception as e:
                logger.warning("Failed to load CW-All membership: %s", e)
        return member_ra, member_dec, bcg_ra, bcg_dec

    # CW-HCG: Py18_Groups_membership.fits
    if catalog_name.strip().lower() in ("cw-hcg", "cw_hcg", "py18"):
        path_fits = data_dir / "Py18_Groups_membership.fits"
        if path_fits.exists():
            try:
                mem = Table.read(path_fits)
                if "Group_ID" not in mem.colnames:
                    return None, None, None, None
                mask = np.array([match_gid(g) for g in mem["Group_ID"]])
                if not np.any(mask):
                    return None, None, None, None
                sub = mem[mask]
                # Require LP_warn_fl == 0 when present (clean galaxies only)
                if "LP_warn_fl" in sub.colnames:
                    try:
                        warn = np.asarray(sub["LP_warn_fl"], dtype=float)
                        sub = sub[np.isfinite(warn) & (warn == 0)]
                    except Exception:
                        pass
                if len(sub) == 0:
                    return None, None, None, None
                # Try RA_MODEL/DEC_MODEL first (as per user's note), then fallback to other column names
                ra_col = None
                dec_col = None
                for col_pair in [("RA_MODEL", "DEC_MODEL"), ("Ra", "Dec"), ("RA", "DEC"), ("gal_Ra", "gal_Dec")]:
                    if col_pair[0] in sub.colnames and col_pair[1] in sub.colnames:
                        ra_col, dec_col = col_pair
                        break
                if ra_col is None or dec_col is None:
                    return None, None, None, None
                member_ra = np.asarray(sub[ra_col], dtype=float)
                member_dec = np.asarray(sub[dec_col], dtype=float)
                # Exclude galaxies with bad/sentinel RA/Dec
                clean = _is_clean_ra_dec(member_ra, member_dec)
                if not np.any(clean):
                    return None, None, None, None
                member_ra = member_ra[clean]
                member_dec = member_dec[clean]
                sub = sub[clean]
                # BCG identification: try LP_mass_med_PDF first, then other mass columns
                mass_col = None
                for col in ["LP_mass_med_PDF", "mass", "MASS", "stellar_mass", "M_stellar"]:
                    if col in sub.colnames:
                        mass_col = col
                        break
                if mass_col is not None:
                    mass = np.asarray(sub[mass_col], dtype=float)
                    valid = np.isfinite(mass)
                    if np.any(valid):
                        idx_bcg = np.nanargmax(np.where(valid, mass, -np.inf))
                        bcg_ra = float(member_ra[idx_bcg])
                        bcg_dec = float(member_dec[idx_bcg])
            except Exception as e:
                logger.warning("Failed to load CW-HCG membership: %s", e)
        return member_ra, member_dec, bcg_ra, bcg_dec

    return None, None, None, None


def find_row_by_group_id(table: Table, target_group_id: str) -> Optional[int]:
    """
    Find table row index for a given Group_ID (match by int or string).

    Returns
    -------
    int or None
        Row index if found, else None.
    """
    gid_col = None
    for col in ["Group_ID", "ID", "group_id", "id", "GroupID"]:
        if col in table.colnames:
            gid_col = col
            break
    if gid_col is None:
        return None
    try:
        target_int = int(target_group_id)
    except (ValueError, TypeError):
        target_int = None
    for i in range(len(table)):
        val = table[i][gid_col]
        if val is None:
            continue
        try:
            if target_int is not None and int(val) == target_int:
                return i
        except (ValueError, TypeError):
            pass
        if str(val).strip() == str(target_group_id).strip():
            return i
    return None


def pick_showcase_row(
    table: Table,
    require_detected: bool = True,
    min_snr: Optional[float] = None,
) -> Optional[int]:
    """
    Pick the best showcase row from the table.

    Parameters
    ----------
    table : Table
        Results table with detection information
    require_detected : bool
        If True, only select from detected groups (default: True)
    min_snr : float, optional
        If set, only consider rows with SNR >= min_snr when selecting by highest SNR.

    Returns
    -------
    int or None
        Index of selected row, or None if no suitable row found
    """
    if "Is_Detected" not in table.colnames:
        if require_detected:
            logger.warning("No 'Is_Detected' column found, but require_detected=True. Cannot select detected group.")
            return None
        else:
            logger.warning("No 'Is_Detected' column found. Selecting highest SNR group.")
            if "SNR" in table.colnames:
                snr_all = np.asarray(table["SNR"], dtype=float)
                if min_snr is not None:
                    snr_all = np.where(snr_all >= min_snr, snr_all, np.nan)
                if np.any(np.isfinite(snr_all)):
                    return int(np.nanargmax(snr_all))
            return 0 if len(table) > 0 else None

    detected = np.asarray(table["Is_Detected"], dtype=bool)
    if min_snr is not None and "SNR" in table.colnames:
        snr_arr = np.asarray(table["SNR"], dtype=float)
        detected = detected & (np.isfinite(snr_arr) & (snr_arr >= min_snr))
    n_detected = np.sum(detected)

    if require_detected:
        if n_detected == 0:
            logger.warning(
                "No detected groups found in table%s. Cannot create showcase.",
                " (with SNR >= %.1f)" % (min_snr,) if min_snr is not None else "",
            )
            return None

        logger.info(
            "Found %d detected groups%s. Selecting highest SNR.",
            n_detected,
            " (SNR >= %.1f)" % (min_snr,) if min_snr is not None else "",
        )
        subset = table[detected]
        snr = np.asarray(subset["SNR"], dtype=float)
        if np.any(np.isfinite(snr)):
            idx = np.nanargmax(snr)
            selected_idx = int(np.where(detected)[0][idx])
            logger.info("Selected detected group at index %d with SNR = %.2f",
                       selected_idx, snr[idx])
            return selected_idx
        else:
            logger.warning("No valid SNR values for detected groups.")
            return int(np.where(detected)[0][0])  # Return first detected group

    # Fallback: if not requiring detected, use highest SNR overall
    if "SNR" in table.colnames:
        snr_all = np.asarray(table["SNR"], dtype=float)
        if min_snr is not None:
            snr_all = np.where(snr_all >= min_snr, snr_all, np.nan)
        if np.any(np.isfinite(snr_all)):
            return int(np.nanargmax(snr_all))

    return 0 if len(table) > 0 else None


def make_showcase_for_catalog(
    catalog_name: str,
    table: Table,
    xray_maps: Dict,
    cosmology,
    output_dir: Path,
    redshift_threshold: Optional[float] = None,
    membership_data_dir: Optional[Path] = None,
    target_group_id: Optional[str] = None,
    min_snr: Optional[float] = None,
) -> None:
    """
    Create showcase image for a catalog.

    Group selection (one of):
    - If target_group_id is set: use that Group_ID (any detection status).
    - Else: pick highest-SNR detected group, optionally with SNR >= min_snr.

    Uses the full (unmasked) X-ray map. If membership_data_dir is set, loads
    group membership and plots member galaxies (+) and the most massive member (BCG).
    """
    if target_group_id is not None:
        row_idx = find_row_by_group_id(table, str(target_group_id).strip())
        if row_idx is None:
            logger.warning("Group_ID '%s' not found in catalog '%s'. Skipping showcase.", target_group_id, catalog_name)
            return
        logger.info("Showcase: using Group_ID = %s (row index %d)", target_group_id, row_idx)
    else:
        row_idx = pick_showcase_row(table, require_detected=True, min_snr=min_snr)
        if row_idx is None:
            logger.warning("No detected groups available to showcase for catalog '%s'", catalog_name)
            return

    row = table[row_idx]
    ra = float(row["RA"])
    dec = float(row["DEC"])
    redshift = float(row["Redshift"])
    is_detected = bool(row.get("Is_Detected", False))
    snr = float(row.get("SNR", np.nan))
    
    # Try to get Group ID from various possible column names
    group_id = None
    for col_name in ["Group_ID", "ID", "group_id", "id", "GroupID"]:
        if col_name in table.colnames:
            group_id_val = row.get(col_name)
            if group_id_val is not None:
                group_id = str(group_id_val)
                break
    
    # Get actual aperture used in analysis
    aperture_kpc_actual = float(row.get("Aperture_kpc", 300.0))
    aperture_arcsec_actual = float(row.get("Aperture_Arcsec", np.nan))
    
    # Get R500 and R200 if available (from mass estimates)
    r500_kpc = float(row.get("R500_kpc", np.nan)) if "R500_kpc" in table.colnames else np.nan
    r200_kpc = float(row.get("R200_kpc", np.nan)) if "R200_kpc" in table.colnames else np.nan
    
    # Get actual background radii used in analysis (from adaptive background if available)
    bg_inner_kpc_actual = float(row.get("Background_Inner_kpc", np.nan)) if "Background_Inner_kpc" in table.colnames else np.nan
    bg_outer_kpc_actual = float(row.get("Background_Outer_kpc", np.nan)) if "Background_Outer_kpc" in table.colnames else np.nan
    
    logger.info("Selected group: RA=%.4f°, Dec=%.4f°, z=%.2f, Detected=%s, SNR=%.2f",
                ra, dec, redshift, is_detected, snr)
    logger.info("  Aperture used: %.1f kpc (%.1f arcsec)", aperture_kpc_actual, aperture_arcsec_actual)
    if np.isfinite(r500_kpc):
        logger.info("  R500 (from mass): %.1f kpc (aperture is %.1f%% of R500)", 
                   r500_kpc, 100 * aperture_kpc_actual / r500_kpc)
    if np.isfinite(r200_kpc):
        logger.info("  R200 (from mass): %.1f kpc", r200_kpc)
    if np.isfinite(bg_inner_kpc_actual) and np.isfinite(bg_outer_kpc_actual):
        logger.info("  Background annulus used: %.1f-%.1f kpc", bg_inner_kpc_actual, bg_outer_kpc_actual)
    
    # Select appropriate X-ray map: use full (unmasked) map for detected groups
    # This shows the complete X-ray emission including extended sources
    if 'full' in xray_maps:
        xray_map = xray_maps['full']
        map_type = "full (unmasked)"
        logger.info("Using %s X-ray map for detected group showcase", map_type)
    elif 'single' in xray_maps:
        xray_map = xray_maps['single']
        map_type = "single"
        logger.info("Using single X-ray map for showcase")
    else:
        logger.error("No suitable X-ray map found. Need 'full' or 'single' map for detected groups.")
        return

    output_path = output_dir / f"showcase_{slugify(catalog_name)}.png"
    # No title - all information is in the legend

    # Load group membership for this group (member positions + BCG)
    member_ra, member_dec, bcg_ra, bcg_dec = None, None, None, None
    if membership_data_dir is not None and group_id is not None:
        member_ra, member_dec, bcg_ra, bcg_dec = load_group_membership(
            catalog_name, group_id, membership_data_dir
        )
        if member_ra is not None:
            logger.info("Loaded %d group members for showcase", len(member_ra))

    # Use actual values from analysis results table (includes adaptive background if applied)
    # Fallback to R500-based calculation if results table values are missing
    if np.isfinite(bg_inner_kpc_actual) and np.isfinite(bg_outer_kpc_actual) and bg_inner_kpc_actual > 0 and bg_outer_kpc_actual > 0:
        # Use actual background radii from results table (reflects adaptive background if used)
        bg_inner_kpc = bg_inner_kpc_actual
        bg_outer_kpc = bg_outer_kpc_actual
        logger.info("  Using background annulus from results table: %.1f-%.1f kpc", bg_inner_kpc, bg_outer_kpc)
    elif is_detected and np.isfinite(r500_kpc) and r500_kpc > 0:
        # Fallback: Use R500-based calculation if results table values are missing
        bg_inner_kpc = r500_kpc * 1.5
        bg_outer_kpc = r500_kpc * 2.0
        logger.info("  Using R500-based background (fallback): %.1f-%.1f kpc", bg_inner_kpc, bg_outer_kpc)
    else:
        # Default fixed values
        bg_inner_kpc = 500.0
        bg_outer_kpc = 600.0
        logger.info("  Using default background annulus: %.1f-%.1f kpc", bg_inner_kpc, bg_outer_kpc)
    
    # Use actual aperture from analysis, or R500 if available and larger
    # For detected groups, R500 is more physically meaningful than fixed 300 kpc
    if is_detected and np.isfinite(r500_kpc) and r500_kpc > 0:
        # Use R500 if it's significantly larger than the fixed aperture
        # This shows the true group radius vs what was used in analysis
        source_radius_kpc = max(aperture_kpc_actual, r500_kpc)
        logger.info("  Using R500=%.1f kpc for showcase (analysis used %.1f kpc)", 
                   r500_kpc, aperture_kpc_actual)
    else:
        # Use actual aperture from analysis
        source_radius_kpc = aperture_kpc_actual
    
    plot_group_showcase(
        xray_map=xray_map,
        ra=ra,
        dec=dec,
        redshift=redshift,
        cosmology=cosmology,
        output_path=str(output_path),
        title=None,  # No title - information is in legend
        source_radius_kpc=source_radius_kpc,
        background_inner_kpc=bg_inner_kpc,
        background_outer_kpc=bg_outer_kpc,
        find_xray_peak=True,  # Find and display X-ray peak position
        use_xray_center=False,  # Keep catalog center, but show X-ray peak offset
        r500_kpc=r500_kpc if np.isfinite(r500_kpc) else None,  # Show R500 reference
        r200_kpc=r200_kpc if np.isfinite(r200_kpc) else None,  # Show R200 reference
        aperture_kpc_actual=aperture_kpc_actual,  # Show what was actually used
        group_id=group_id,  # Group ID for legend
        snr=snr if np.isfinite(snr) else None,  # SNR for legend
        xray_map_label=map_type,  # Show which X-ray map is used (e.g. "full (unmasked)")
        member_ra=member_ra,  # Group member positions (plotted as +)
        member_dec=member_dec,
        bcg_ra=bcg_ra,  # Most massive member (plotted with distinct marker)
        bcg_dec=bcg_dec,
    )


def load_results_table(results_dir: Path) -> Table:
    path = results_dir / "xray_catalog.fits"
    if not path.exists():
        raise FileNotFoundError(f"Results table not found: {path}")
    return Table.read(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate zoomed X-ray showcase images for COSMOS-Web groups."
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
        "--group-id",
        type=str,
        default=None,
        metavar="ID",
        help="Showcase this Group_ID (overrides default: highest-SNR detected). Applied to each catalog.",
    )
    parser.add_argument(
        "--min-snr",
        type=float,
        default=None,
        metavar="S",
        help="When selecting by SNR, only consider groups with SNR >= S (ignored if --group-id is set).",
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

    # Load X-ray maps (supports both single and dual map modes)
    # For detected groups, we'll use the full (unmasked) map
    xray_maps = load_xray_maps_from_config(config)
    
    if 'full' not in xray_maps and 'single' not in xray_maps:
        logger.error("No 'full' or 'single' X-ray map found. Need unmasked map for detected groups.")
        raise ValueError("X-ray map configuration must include 'full' map for detected group showcase.")

    output_base = Path(config["output"]["figures_dir"]).resolve()

    catalogs = config.get("data", {}).get("catalogs", [])
    if not catalogs:
        raise ValueError("No catalogs defined in configuration.")

    requested = set(args.catalog) if args.catalog else None

    for entry in catalogs:
        name = entry.get("name")
        if not name:
            continue
        if requested and name not in requested:
            continue

        slug = slugify(name)
        results_dir = Path(config["output"]["results_dir"]) / slug
        figures_dir = output_base / slug
        figures_dir.mkdir(parents=True, exist_ok=True)

        try:
            table = load_results_table(results_dir)
        except FileNotFoundError as err:
            logger.warning("Skipping %s: %s", name, err)
            continue

        # Get catalog-specific redshift threshold if available
        catalog_threshold = entry.get('redshift_threshold', None)
        
        # Membership files live next to the group catalog
        membership_data_dir = Path(entry["group_catalog"]).parent if entry.get("group_catalog") else None
        logger.info(
            "Creating showcase image for catalog '%s'%s",
            name,
            " (Group_ID=%s)" % args.group_id if args.group_id else " (highest SNR detected)" + (" SNR>=%.1f" % args.min_snr if args.min_snr is not None else ""),
        )
        make_showcase_for_catalog(
            catalog_name=name,
            table=table,
            xray_maps=xray_maps,
            cosmology=cosmology,
            output_dir=figures_dir,
            redshift_threshold=catalog_threshold,
            membership_data_dir=membership_data_dir,
            target_group_id=args.group_id,
            min_snr=args.min_snr,
        )


if __name__ == "__main__":
    main()
