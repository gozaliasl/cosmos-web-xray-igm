#!/usr/bin/env python3
"""Quick script to examine the data files."""

from astropy.io import fits
import numpy as np

print("=" * 60)
print("GROUP CATALOG")
print("=" * 60)

with fits.open('/Users/gozalig1/Projects/compact-groups-xray-analysis/data/group-catalog/Py18_Groups.fits') as hdul:
    hdul.info()
    data = hdul[1].data
    cols = hdul[1].columns.names

    print(f"\nNumber of groups: {len(data)}")
    print(f"\nColumn names ({len(cols)} total):")
    for i, col in enumerate(cols):
        print(f"  {i+1:2d}. {col}")

    print("\nFirst 3 groups (selected columns):")
    key_cols = ['Ra', 'Dec', 'z', 'dz','Grp']
    available_key_cols = [c for c in key_cols if c in cols]
    for col in available_key_cols:
        print(f"  {col}: {data[col][:3]}")

print("\n" + "=" * 60)
print("X-RAY MAP")
print("=" * 60)

with fits.open('/Users/gozalig1/Projects/compact-groups-xray-analysis/data/xray-map/cosmos_chaxmm14_noem_520.fits') as hdul:
    hdul.info()
    header = hdul[0].header
    data = hdul[0].data

    print(f"\nImage shape: {data.shape}")
    print(f"Data range: [{np.nanmin(data):.6e}, {np.nanmax(data):.6e}]")
    print(f"Non-NaN pixels: {np.sum(~np.isnan(data))}")

    print("\nWCS Header keywords:")
    wcs_keys = ['NAXIS1', 'NAXIS2', 'CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2',
                'CD1_1', 'CD2_2', 'CDELT1', 'CDELT2', 'CTYPE1', 'CTYPE2', 'CUNIT1', 'CUNIT2']
    for key in wcs_keys:
        if key in header:
            print(f"  {key}: {header[key]}")

print("\n" + "=" * 60)
print("X-RAY ERROR MAP")
print("=" * 60)

with fits.open('/Users/gozalig1/Projects/compact-groups-xray-analysis/data/xray-map/cosmos_chaxmm14_noem_520_err.fits') as hdul:
    hdul.info()
    data = hdul[0].data

    print(f"\nError map shape: {data.shape}")
    print(f"Error range: [{np.nanmin(data):.6e}, {np.nanmax(data):.6e}]")
    print(f"Non-NaN pixels: {np.sum(~np.isnan(data))}")
