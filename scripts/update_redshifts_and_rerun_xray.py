#!/usr/bin/env python3
"""
Update group catalogs with VRF-refined redshifts and re-run X-ray analysis.

This script:
1. Loads refined redshift summaries (z_vrf_refined from membership analysis)
2. Updates the original group catalogs (FITS) with refined redshifts
3. Saves updated catalogs (with _refined_z suffix)
4. Optionally re-runs main_analysis.py and stacking with updated redshifts

Usage:
    python update_redshifts_and_rerun_xray.py [--rerun-analysis] [--rerun-stacking]
"""

import sys
import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.io import fits

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent
GROUP_CATALOG_DIR = BASE_DIR / 'data' / 'group-catalog'
REFINED_REDSHIFT_DIR = BASE_DIR / 'membership_determination' / 'results' / 'specz'

# Original catalogs (FITS)
CW_ALL_CATALOG = GROUP_CATALOG_DIR / 'cosmos_web_groups_catalog.fits'
CW_HCG_CATALOG = GROUP_CATALOG_DIR / 'Py18_Groups.fits'

# Refined redshift summaries
CW_ALL_REFINED = REFINED_REDSHIFT_DIR / 'cw_all_summary_vrf_rhybrid.csv'
CW_HCG_REFINED = REFINED_REDSHIFT_DIR / 'cw_hcg_summary_vrf_rhybrid.csv'


def load_refined_redshifts(summary_path: Path) -> pd.DataFrame:
    """Load refined redshift summary."""
    if not summary_path.exists():
        raise FileNotFoundError(f"Refined redshift summary not found: {summary_path}")
    df = pd.read_csv(summary_path)
    logger.info(f"Loaded {len(df)} groups from {summary_path.name}")
    return df


def update_catalog_with_refined_z(
    catalog_path: Path,
    refined_df: pd.DataFrame,
    group_id_col: str = 'Group_ID',
    output_suffix: str = '_refined_z'
) -> Path:
    """
    Update FITS catalog with refined redshifts.
    
    Parameters
    ----------
    catalog_path : Path
        Path to original FITS catalog
    refined_df : DataFrame
        DataFrame with Group_ID and z_vrf_refined columns
    group_id_col : str
        Column name for group ID in refined_df (default: 'Group_ID')
    output_suffix : str
        Suffix for output filename (default: '_refined_z')
    
    Returns
    -------
    Path
        Path to updated catalog
    """
    logger.info(f"\nUpdating catalog: {catalog_path.name}")
    
    # Load original FITS catalog
    with fits.open(catalog_path) as hdul:
        data = Table(hdul[1].data)
        header = hdul[0].header.copy() if len(hdul) > 0 else None
    
    # Find redshift column (z or Redshift)
    z_col = None
    for col in ['z', 'Redshift']:
        if col in data.colnames:
            z_col = col
            break
    
    if z_col is None:
        raise ValueError(f"No redshift column (z/Redshift) found in {catalog_path}")
    
    # Find Group_ID column in catalog
    catalog_id_col = None
    for col in ['Group_ID', 'ID', 'group_id', 'id', 'GroupID', 'GROUP_ID', 'Grp', 'GrpID']:
        if col in data.colnames:
            catalog_id_col = col
            break
    
    if catalog_id_col is None:
        raise ValueError(f"No Group ID column found in {catalog_path}")
    
    logger.info(f"  Using ID column: {catalog_id_col}, redshift column: {z_col}")
    
    # Convert IDs to strings for matching
    catalog_ids = pd.Series(data[catalog_id_col]).astype(str)
    refined_ids = pd.Series(refined_df[group_id_col]).astype(str)
    
    # Match groups
    match_mask = catalog_ids.isin(refined_ids)
    n_matched = match_mask.sum()
    logger.info(f"  Matched {n_matched}/{len(data)} groups")
    
    # Create updated redshift array
    z_original = np.array(data[z_col], dtype=float)
    z_updated = z_original.copy()
    
    # Update matched groups with refined redshifts
    n_updated = 0
    n_kept_original = 0
    
    for idx in np.where(match_mask)[0]:
        catalog_id_str = str(data[catalog_id_col][idx])
        refined_row = refined_df[refined_df[group_id_col].astype(str) == catalog_id_str]
        
        if len(refined_row) > 0:
            z_refined = refined_row['z_vrf_refined'].iloc[0]
            z_original_val = z_original[idx]
            
            # Use refined redshift if valid, otherwise keep original
            if pd.notna(z_refined) and np.isfinite(z_refined):
                z_updated[idx] = z_refined
                n_updated += 1
            else:
                # Keep original redshift (photo-z) if no spec-z members
                n_kept_original += 1
    
    logger.info(f"  Updated redshifts: {n_updated} (refined), {n_kept_original} (kept original/photo-z)")
    logger.info(f"  Redshift range: z = [{np.nanmin(z_updated):.3f}, {np.nanmax(z_updated):.3f}]")
    
    # Update data table
    data[z_col] = z_updated
    
    # Add metadata column indicating which groups were updated
    if 'z_refined_source' not in data.colnames:
        z_source = np.full(len(data), 'original', dtype='U20')
        z_source[match_mask] = 'vrf_refined'
        data['z_refined_source'] = z_source
    
    # Save updated catalog
    output_path = catalog_path.parent / f"{catalog_path.stem}{output_suffix}.fits"
    
    # Create new FITS file
    hdu_list = fits.HDUList()
    if header is not None:
        hdu_list.append(fits.PrimaryHDU(header=header))
    else:
        hdu_list.append(fits.PrimaryHDU())
    
    hdu_list.append(fits.BinTableHDU(data))
    hdu_list.writeto(output_path, overwrite=True)
    
    logger.info(f"  Saved updated catalog: {output_path}")
    
    return output_path


def update_config_for_refined_catalogs(config_path: Path, output_catalogs: dict) -> Path:
    """
    Create updated config.yaml pointing to refined-z catalogs.
    
    Parameters
    ----------
    config_path : Path
        Path to original config.yaml
    output_catalogs : dict
        Dict mapping catalog names to updated catalog paths
    
    Returns
    -------
    Path
        Path to updated config file
    """
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update catalog paths
    if 'catalogs' in config.get('data', {}):
        for entry in config['data']['catalogs']:
            name = entry.get('name', '')
            if name in output_catalogs:
                entry['group_catalog'] = str(output_catalogs[name])
                logger.info(f"Updated config: {name} -> {output_catalogs[name].name}")
    
    # Save updated config
    output_config = config_path.parent / f"{config_path.stem}_refined_z.yaml"
    with open(output_config, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Saved updated config: {output_config}")
    return output_config


def main():
    parser = argparse.ArgumentParser(
        description='Update group catalogs with VRF-refined redshifts and optionally re-run X-ray analysis'
    )
    parser.add_argument('--rerun-analysis', action='store_true',
                       help='Re-run main_analysis.py with updated redshifts')
    parser.add_argument('--rerun-stacking', action='store_true',
                       help='Re-run stacking analysis (requires --rerun-analysis or existing results)')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config.yaml (default: config.yaml)')
    
    args = parser.parse_args()
    
    logger.info("="*70)
    logger.info("UPDATING GROUP CATALOGS WITH VRF-REFINED REDSHIFTS")
    logger.info("="*70)
    
    # Load refined redshift summaries
    logger.info("\nLoading refined redshift summaries...")
    cw_all_refined = load_refined_redshifts(CW_ALL_REFINED)
    cw_hcg_refined = load_refined_redshifts(CW_HCG_REFINED)
    
    # Update catalogs
    logger.info("\n" + "="*70)
    logger.info("UPDATING CATALOGS")
    logger.info("="*70)
    
    output_catalogs = {}
    
    # CW-All
    if CW_ALL_CATALOG.exists():
        updated_all = update_catalog_with_refined_z(
            CW_ALL_CATALOG, cw_all_refined, group_id_col='Group_ID'
        )
        output_catalogs['CW-All'] = updated_all
    else:
        logger.warning(f"CW-All catalog not found: {CW_ALL_CATALOG}")
    
    # CW-HCG
    if CW_HCG_CATALOG.exists():
        updated_hcg = update_catalog_with_refined_z(
            CW_HCG_CATALOG, cw_hcg_refined, group_id_col='Group_ID'
        )
        output_catalogs['CW-HCG'] = updated_hcg
    else:
        logger.warning(f"CW-HCG catalog not found: {CW_HCG_CATALOG}")
    
    # Update config.yaml
    config_path = Path(args.config)
    if config_path.exists():
        logger.info("\n" + "="*70)
        logger.info("UPDATING CONFIG")
        logger.info("="*70)
        updated_config = update_config_for_refined_catalogs(config_path, output_catalogs)
    else:
        logger.warning(f"Config file not found: {config_path}")
        updated_config = None
    
    # Re-run analysis if requested
    if args.rerun_analysis:
        logger.info("\n" + "="*70)
        logger.info("RE-RUNNING X-RAY ANALYSIS")
        logger.info("="*70)
        
        if updated_config is None:
            logger.error("Cannot re-run analysis: updated config not created")
            sys.exit(1)
        
        import subprocess
        cmd = ['python', 'main_analysis.py', '--config', str(updated_config)]
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=BASE_DIR)
        
        if result.returncode != 0:
            logger.error("X-ray analysis failed")
            sys.exit(1)
        
        logger.info("X-ray analysis complete")
    
    # Note: Stacking is included in main_analysis.py, so --rerun-analysis already re-runs stacking
    # After re-running analysis, you can run analyze_stacking.py separately if needed
    if args.rerun_stacking:
        logger.info("\n" + "="*70)
        logger.info("NOTE: Stacking is included in main_analysis.py")
        logger.info("Stacking results are automatically updated when --rerun-analysis is used")
        logger.info("To analyze stacking results, run:")
        logger.info("  python analyze_stacking.py --config config_refined_z.yaml --catalog CW-All")
        logger.info("  python analyze_stacking.py --config config_refined_z.yaml --catalog CW-HCG")
        logger.info("="*70)
    
    logger.info("\n" + "="*70)
    logger.info("COMPLETE")
    logger.info("="*70)
    logger.info("\nUpdated catalogs:")
    for name, path in output_catalogs.items():
        logger.info(f"  {name}: {path}")
    if updated_config:
        logger.info(f"\nUpdated config: {updated_config}")
        logger.info(f"\nTo re-run analysis manually:")
        logger.info(f"  python main_analysis.py --config {updated_config}")


if __name__ == '__main__':
    main()
