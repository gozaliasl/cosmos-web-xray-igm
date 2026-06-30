#!/usr/bin/env python3
"""
Check Background_Valid_Pixels values in X-ray catalog results.
"""

from astropy.table import Table
import numpy as np
from pathlib import Path
import argparse


def check_catalog(catalog_path: Path):
    """Check Background_Valid_Pixels in a catalog."""
    print(f"\n{'='*70}")
    print(f"Checking: {catalog_path}")
    print(f"{'='*70}")
    
    if not catalog_path.exists():
        print(f"❌ File not found: {catalog_path}")
        return
    
    t = Table.read(catalog_path)
    
    # Check if column exists
    if 'Background_Valid_Pixels' not in t.colnames:
        print("❌ Column 'Background_Valid_Pixels' does NOT exist!")
        bg_cols = [c for c in t.colnames if 'Background' in c]
        print(f"Available Background columns: {bg_cols}")
        return
    
    print("✓ Column 'Background_Valid_Pixels' exists")
    
    # Get values
    col = t['Background_Valid_Pixels']
    if hasattr(col, 'filled'):
        values = np.array(col.filled(np.nan))
    else:
        values = np.array(col)
    
    # Statistics
    finite_mask = np.isfinite(values)
    non_zero_mask = (values > 0) & finite_mask
    
    print(f"\nTotal groups: {len(t)}")
    print(f"Groups with valid pixel counts: {np.sum(finite_mask)} ({np.sum(finite_mask)/len(t)*100:.1f}%)")
    print(f"Groups with non-zero pixel counts: {np.sum(non_zero_mask)} ({np.sum(non_zero_mask)/len(t)*100:.1f}%)")
    print(f"Groups with NaN/masked values: {np.sum(~finite_mask)} ({np.sum(~finite_mask)/len(t)*100:.1f}%)")
    
    if np.any(finite_mask):
        finite_vals = values[finite_mask]
        print(f"\nValid Pixel Counts Statistics:")
        print(f"  Min: {np.min(finite_vals):.0f}")
        print(f"  Max: {np.max(finite_vals):.0f}")
        print(f"  Median: {np.median(finite_vals):.0f}")
        print(f"  Mean: {np.mean(finite_vals):.0f}")
        print(f"  25th percentile: {np.percentile(finite_vals, 25):.0f}")
        print(f"  75th percentile: {np.percentile(finite_vals, 75):.0f}")
        
        # Show examples
        print(f"\nFirst 10 groups with valid pixel counts:")
        valid_indices = np.where(finite_mask)[0][:10]
        for idx in valid_indices:
            group_id = t['Group_ID'][idx] if 'Group_ID' in t.colnames else idx
            val = values[idx]
            detected = t['Is_Detected'][idx] if 'Is_Detected' in t.colnames else 'N/A'
            print(f"  Group {group_id}: {val:.0f} pixels (Detected: {detected})")
    else:
        print("\n⚠️  WARNING: No groups have valid pixel counts!")
        print("   All values are NaN/masked.")
        print("   This suggests the background measurement code needs to be fixed.")
        
        # Check a few groups to see what's happening
        print(f"\nChecking first 5 groups:")
        for idx in range(min(5, len(t))):
            group_id = t['Group_ID'][idx] if 'Group_ID' in t.colnames else idx
            val = values[idx]
            bg = t['Background'][idx] if 'Background' in t.colnames else np.nan
            detected = t['Is_Detected'][idx] if 'Is_Detected' in t.colnames else 'N/A'
            print(f"  Group {group_id}: ValidPixels={val}, Background={bg:.6f}, Detected={detected}")
    
    # Check detected groups specifically
    if 'Is_Detected' in t.colnames:
        detected = t[t['Is_Detected'] == True]
        if len(detected) > 0:
            det_col = detected['Background_Valid_Pixels']
            if hasattr(det_col, 'filled'):
                det_values = np.array(det_col.filled(np.nan))
            else:
                det_values = np.array(det_col)
            det_finite = np.isfinite(det_values)
            
            print(f"\nDetected Groups ({len(detected)} total):")
            print(f"  With valid pixel counts: {np.sum(det_finite)} / {len(detected)} ({np.sum(det_finite)/len(detected)*100:.1f}%)")
            if np.any(det_finite):
                print(f"  Median valid pixels: {np.median(det_values[det_finite]):.0f}")
                print(f"  Mean valid pixels: {np.mean(det_values[det_finite]):.0f}")
                print(f"  Range: {np.min(det_values[det_finite]):.0f} - {np.max(det_values[det_finite]):.0f}")
    
    # Check Group 1 specifically
    if 'Group_ID' in t.colnames:
        try:
            row_idx = np.where(t['Group_ID'] == 1)[0]
            if len(row_idx) > 0:
                row = t[row_idx[0]]
                val = values[row_idx[0]]
                print(f"\nGroup 1 Details:")
                print(f"  Background_Valid_Pixels: {val}")
                print(f"  Is finite: {np.isfinite(val) if isinstance(val, (int, float, np.number)) else 'N/A'}")
                if 'Background' in t.colnames:
                    print(f"  Background: {row['Background']:.6f}")
                if 'SNR' in t.colnames:
                    print(f"  SNR: {row['SNR']:.2f}")
                if 'Background_Inner_kpc' in t.colnames:
                    print(f"  Background annulus: {row['Background_Inner_kpc']:.1f} - {row['Background_Outer_kpc']:.1f} kpc")
        except Exception as e:
            print(f"\nGroup 1: Error checking - {e}")


def main():
    parser = argparse.ArgumentParser(description="Check Background_Valid_Pixels in X-ray catalog")
    parser.add_argument('--catalog', type=str, default='cw_all',
                       help='Catalog slug (cw_all or cw_hcg)')
    parser.add_argument('--results-dir', type=Path, default=Path('outputs/results'),
                       help='Results directory')
    
    args = parser.parse_args()
    
    catalog_path = args.results_dir / args.catalog / "xray_catalog.fits"
    check_catalog(catalog_path)
    
    # Also check other catalog if exists
    other_catalog = 'cw_hcg' if args.catalog == 'cw_all' else 'cw_all'
    other_path = args.results_dir / other_catalog / "xray_catalog.fits"
    if other_path.exists():
        check_catalog(other_path)


if __name__ == '__main__':
    main()
