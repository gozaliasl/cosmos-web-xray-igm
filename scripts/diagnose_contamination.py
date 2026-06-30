#!/usr/bin/env python3
"""
Diagnostic script to analyze contamination and determine if high-z groups
are real detections or just background from low-z groups.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.table import Table
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_contamination_diagnosis(
    catalog_slug: str,
    results_dir: Path,
    output_dir: Path
):
    """Analyze contaminated vs non-contaminated groups to diagnose the issue."""
    
    catalog_path = results_dir / catalog_slug / "xray_catalog.fits"
    if not catalog_path.exists():
        catalog_path = results_dir / catalog_slug / "xray_catalog.csv"
    
    if not catalog_path.exists():
        logger.error(f"Catalog not found: {catalog_path}")
        return
    
    table = Table.read(catalog_path)
    
    if 'Is_Projected_Contaminated' not in table.colnames:
        logger.error("No contamination data. Run analysis with check_projected_contamination: true")
        return
    
    redshift = np.array(table['Redshift'], dtype=float)
    contaminated = np.array(table['Is_Projected_Contaminated'], dtype=float)
    is_detected = np.array(table['Is_Detected'], dtype=bool)
    snr = np.array(table['SNR'], dtype=float)
    background = np.array(table['Background'], dtype=float)
    net_counts = np.array(table['Net_Counts'], dtype=float)
    luminosity = np.array(table['Luminosity_erg_s'], dtype=float)
    m200 = np.array(table['M200_Luminosity_Msun'], dtype=float) if 'M200_Luminosity_Msun' in table.colnames else None
    
    # Filter high-z groups
    high_z_mask = redshift >= 1.5
    high_z_detected = high_z_mask & is_detected
    
    contaminated_high_z = high_z_detected & (contaminated > 0)
    clean_high_z = high_z_detected & (contaminated == 0)
    
    print("\n" + "="*70)
    print(f"CONTAMINATION DIAGNOSIS: {catalog_slug.upper()}")
    print("="*70)
    print(f"\nHigh-z groups (z ≥ 1.5): {np.sum(high_z_mask)}")
    print(f"High-z detected: {np.sum(high_z_detected)}")
    print(f"  - Contaminated: {np.sum(contaminated_high_z)} ({100*np.sum(contaminated_high_z)/np.sum(high_z_detected):.1f}%)")
    print(f"  - Clean: {np.sum(clean_high_z)} ({100*np.sum(clean_high_z)/np.sum(high_z_detected):.1f}%)")
    
    if np.sum(contaminated_high_z) == 0:
        print("\nNo contaminated high-z groups found!")
        return
    
    # Compare properties
    print("\n" + "-"*70)
    print("COMPARISON: Contaminated vs Clean High-z Groups")
    print("-"*70)
    
    # Background levels
    bg_contaminated = background[contaminated_high_z]
    bg_clean = background[clean_high_z]
    
    if len(bg_contaminated) > 0 and len(bg_clean) > 0:
        print(f"\nBackground levels:")
        print(f"  Contaminated: median={np.median(bg_contaminated):.4e}, mean={np.mean(bg_contaminated):.4e}")
        print(f"  Clean:        median={np.median(bg_clean):.4e}, mean={np.mean(bg_clean):.4e}")
        print(f"  Ratio:        {np.median(bg_contaminated)/np.median(bg_clean):.2f}×")
    
    # SNR
    snr_contaminated = snr[contaminated_high_z]
    snr_clean = snr[clean_high_z]
    
    if len(snr_contaminated) > 0 and len(snr_clean) > 0:
        print(f"\nSNR:")
        print(f"  Contaminated: median={np.median(snr_contaminated):.2f}, mean={np.mean(snr_contaminated):.2f}")
        print(f"  Clean:        median={np.median(snr_clean):.2f}, mean={np.mean(snr_clean):.2f}")
    
    # Luminosity
    lum_contaminated = luminosity[contaminated_high_z]
    lum_clean = luminosity[clean_high_z]
    
    if len(lum_contaminated) > 0 and len(lum_clean) > 0:
        valid_cont = np.isfinite(lum_contaminated) & (lum_contaminated > 0)
        valid_clean = np.isfinite(lum_clean) & (lum_clean > 0)
        if np.sum(valid_cont) > 0 and np.sum(valid_clean) > 0:
            print(f"\nLuminosity:")
            print(f"  Contaminated: median={np.median(lum_contaminated[valid_cont]):.2e}, mean={np.mean(lum_contaminated[valid_cont]):.2e}")
            print(f"  Clean:        median={np.median(lum_clean[valid_clean]):.2e}, mean={np.mean(lum_clean[valid_clean]):.2e}")
            print(f"  Ratio:        {np.median(lum_contaminated[valid_cont])/np.median(lum_clean[valid_clean]):.2f}×")
    
    # Check if contaminated groups have elevated background (key diagnostic)
    if len(bg_contaminated) > 0 and len(bg_clean) > 0:
        bg_ratio = np.median(bg_contaminated) / np.median(bg_clean)
        if bg_ratio > 1.5:
            print(f"\n⚠️  WARNING: Contaminated groups have {bg_ratio:.1f}× higher background!")
            print("   This suggests they are measuring background from low-z groups, not real emission.")
        else:
            print(f"\n✓ Background levels similar ({bg_ratio:.2f}×). Contamination may be less severe.")
    
    # Check redshift-binned median background if available
    if 'Background_Binned' in table.colnames:
        bg_binned = np.array(table['Background_Binned'], dtype=float)
        bg_binned_contaminated = bg_binned[contaminated_high_z]
        bg_binned_clean = bg_binned[clean_high_z]
        
        if len(bg_binned_contaminated) > 0 and len(bg_binned_clean) > 0:
            valid_binned_cont = np.isfinite(bg_binned_contaminated) & (bg_binned_contaminated > 0)
            valid_binned_clean = np.isfinite(bg_binned_clean) & (bg_binned_clean > 0)
            
            if np.sum(valid_binned_cont) > 0 and np.sum(valid_binned_clean) > 0:
                print(f"\nRedshift-binned median background:")
                print(f"  Contaminated: median={np.median(bg_binned_contaminated[valid_binned_cont]):.4e}")
                print(f"  Clean:        median={np.median(bg_binned_clean[valid_binned_clean]):.4e}")
                
                # Compare annulus vs binned background for contaminated groups
                bg_annulus_cont = background[contaminated_high_z][valid_binned_cont]
                bg_binned_cont = bg_binned_contaminated[valid_binned_cont]
                ratio_annulus_binned = np.median(bg_annulus_cont) / np.median(bg_binned_cont)
                
                print(f"\n  Annulus/Binned ratio (contaminated): {ratio_annulus_binned:.2f}")
                if ratio_annulus_binned > 1.5:
                    print("  ⚠️  Annulus background >> Binned background!")
                    print("     This confirms contamination: annulus is picking up low-z emission.")
    
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description="Diagnose contamination issues")
    parser.add_argument('--results-dir', type=str, default='outputs/results')
    parser.add_argument('--output-dir', type=str, default='outputs/figures')
    parser.add_argument('--catalog', type=str, choices=['cw_all', 'cw_hcg', 'both'], default='both')
    
    args = parser.parse_args()
    
    catalogs = ['cw_all', 'cw_hcg'] if args.catalog == 'both' else [args.catalog]
    
    for catalog_slug in catalogs:
        analyze_contamination_diagnosis(
            catalog_slug=catalog_slug,
            results_dir=Path(args.results_dir),
            output_dir=Path(args.output_dir)
        )


if __name__ == '__main__':
    main()
