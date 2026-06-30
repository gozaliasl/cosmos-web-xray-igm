#!/usr/bin/env python3
"""
Quick verification script to check if background values in results table
match expected adaptive R500-scaled values.

This helps verify that the adaptive background fix is working correctly.
"""

import numpy as np
from astropy.table import Table
from pathlib import Path
import argparse
import yaml
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u


def load_config(config_path: Path) -> dict:
    """Load configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def verify_background_values(
    results_path: Path,
    config_path: Path,
    catalog_name: str = None
):
    """
    Verify that background values match expected adaptive R500-scaled values.
    
    For detected groups with R500:
    - Expected inner = background_inner_factor_r500 * R500
    - Expected outer = background_outer_factor_r500 * R500
    """
    results = Table.read(results_path)
    config = load_config(config_path)
    
    bg_inner_factor = config.get('photometry', {}).get('background_inner_factor_r500', 2.0)
    bg_outer_factor = config.get('photometry', {}).get('background_outer_factor_r500', 3.5)
    bg_radius_mode = config.get('photometry', {}).get('background_radius_mode', 'fixed_kpc')
    
    print(f"\n{'='*70}")
    print(f"Background Verification")
    print(f"{'='*70}")
    print(f"Config: background_radius_mode = {bg_radius_mode}")
    print(f"Config: background_inner_factor_r500 = {bg_inner_factor}")
    print(f"Config: background_outer_factor_r500 = {bg_outer_factor}")
    print(f"\nTotal groups in results: {len(results)}")
    
    if bg_radius_mode != 'adaptive_r500':
        print(f"\n⚠️  WARNING: background_radius_mode is '{bg_radius_mode}', not 'adaptive_r500'")
        print("   Adaptive background may not be applied.")
        return
    
    # Filter to detected groups with R500
    detected = results['Is_Detected'] == True
    has_r500_col = 'R500_kpc' in results.colnames
    if has_r500_col:
        has_r500 = np.isfinite(results['R500_kpc']) & (results['R500_kpc'] > 0)
    else:
        has_r500 = np.zeros(len(results), dtype=bool)
    
    if not has_r500_col:
        print("\n⚠️  No R500_kpc column found in results. Cannot verify adaptive background.")
        return
    
    valid = detected & has_r500
    n_valid = np.sum(valid)
    print(f"\nDetected groups with R500: {n_valid}")
    
    if n_valid == 0:
        print("No groups to verify.")
        return
    
    # Check background values
    bg_inner_kpc = results['Background_Inner_kpc'][valid]
    bg_outer_kpc = results['Background_Outer_kpc'][valid]
    r500_kpc = results['R500_kpc'][valid]
    
    # Calculate expected values
    expected_inner = bg_inner_factor * r500_kpc
    expected_outer = bg_outer_factor * r500_kpc
    
    # Compare
    inner_diff = np.abs(bg_inner_kpc - expected_inner) / expected_inner
    outer_diff = np.abs(bg_outer_kpc - expected_outer) / expected_outer
    
    # Tolerance: 5% difference (allows for rounding, gap enforcement, etc.)
    tolerance = 0.05
    inner_match = inner_diff < tolerance
    outer_match = outer_diff < tolerance
    
    n_inner_match = np.sum(inner_match)
    n_outer_match = np.sum(outer_match)
    
    print(f"\n{'='*70}")
    print(f"Verification Results")
    print(f"{'='*70}")
    print(f"Groups with matching inner radius (within {tolerance*100:.0f}%): {n_inner_match}/{n_valid} ({n_inner_match/n_valid*100:.1f}%)")
    print(f"Groups with matching outer radius (within {tolerance*100:.0f}%): {n_outer_match}/{n_valid} ({n_outer_match/n_valid*100:.1f}%)")
    
    if n_inner_match < n_valid or n_outer_match < n_valid:
        print(f"\n⚠️  Some groups have mismatched background values:")
        print(f"\n{'Group ID':<15} {'R500':<10} {'BG Inner (actual)':<20} {'BG Inner (expected)':<20} {'Diff %':<10}")
        print(f"{'-'*75}")
        
        mismatched = ~(inner_match & outer_match)
        for idx in np.where(valid)[0][mismatched][:10]:  # Show first 10 mismatched
            group_id = str(results['Group_ID'][idx]) if 'Group_ID' in results.colnames else str(idx)
            r500 = r500_kpc[mismatched][np.where(valid)[0][mismatched] == idx][0] if np.any(mismatched) else np.nan
            bg_inner_act = bg_inner_kpc[mismatched][np.where(valid)[0][mismatched] == idx][0] if np.any(mismatched) else np.nan
            bg_inner_exp = expected_inner[mismatched][np.where(valid)[0][mismatched] == idx][0] if np.any(mismatched) else np.nan
            diff_pct = (bg_inner_act - bg_inner_exp) / bg_inner_exp * 100 if bg_inner_exp > 0 else np.nan
            print(f"{group_id:<15} {r500:<10.1f} {bg_inner_act:<20.1f} {bg_inner_exp:<20.1f} {diff_pct:<10.1f}")
        
        if np.sum(mismatched) > 10:
            print(f"\n... and {np.sum(mismatched) - 10} more")
    else:
        print(f"\n✅ All groups have matching background values!")
    
    # Show example for first few groups
    print(f"\n{'='*70}")
    print(f"Example Groups (first 5)")
    print(f"{'='*70}")
    print(f"{'Group ID':<15} {'R500':<10} {'BG Inner':<15} {'BG Outer':<15} {'Inner/R500':<12} {'Outer/R500':<12}")
    print(f"{'-'*80}")
    for idx in np.where(valid)[0][:5]:
        group_id = str(results['Group_ID'][idx]) if 'Group_ID' in results.colnames else str(idx)
        r500 = r500_kpc[np.where(valid)[0] == idx][0]
        bg_inner = bg_inner_kpc[np.where(valid)[0] == idx][0]
        bg_outer = bg_outer_kpc[np.where(valid)[0] == idx][0]
        inner_ratio = bg_inner / r500 if r500 > 0 else np.nan
        outer_ratio = bg_outer / r500 if r500 > 0 else np.nan
        print(f"{group_id:<15} {r500:<10.1f} {bg_inner:<15.1f} {bg_outer:<15.1f} {inner_ratio:<12.2f} {outer_ratio:<12.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Verify adaptive background values in results table"
    )
    parser.add_argument(
        '--results',
        type=Path,
        required=True,
        help='Path to results xray_catalog.fits file'
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('config.yaml'),
        help='Path to config.yaml file'
    )
    parser.add_argument(
        '--catalog',
        type=str,
        default=None,
        help='Catalog name (for display only)'
    )
    
    args = parser.parse_args()
    
    verify_background_values(
        results_path=args.results,
        config_path=args.config,
        catalog_name=args.catalog
    )


if __name__ == '__main__':
    main()
