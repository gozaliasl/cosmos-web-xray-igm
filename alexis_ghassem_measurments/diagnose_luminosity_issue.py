#!/usr/bin/env python3
"""
Diagnostic script to identify the luminosity underestimation issue.
"""

import pandas as pd
import numpy as np
from astropy.cosmology import Planck18, FlatLambdaCDM
from astropy import units as u

# Load data
csv_data = pd.read_csv('xray_catalog.csv')
fits_data = pd.read_csv('alexis_ghassem.csv', usecols=['z', 'Lx', 'Flux'])

# Match first few sources
print("=" * 80)
print("LUMINOSITY UNDERESTIMATION DIAGNOSTIC")
print("=" * 80)

# Sample a few sources for detailed analysis
n_samples = 5
print(f"\nAnalyzing first {n_samples} matched sources:\n")

for i in range(min(n_samples, len(csv_data), len(fits_data))):
    z = csv_data.iloc[i]['Redshift']
    flux_csv = csv_data.iloc[i]['Flux_erg_cm2_s']
    lum_csv = csv_data.iloc[i]['Luminosity_erg_s']
    
    # Find matching FITS source (by redshift)
    z_diff = np.abs(fits_data['z'] - z)
    fits_idx = np.argmin(z_diff)
    if z_diff.iloc[fits_idx] > 0.01:
        continue
    
    flux_fits = fits_data.iloc[fits_idx]['Flux']
    lum_fits = fits_data.iloc[fits_idx]['Lx']
    
    print(f"Source {i+1}: z = {z:.3f}")
    print(f"  Flux (CSV):  {flux_csv:.3e} erg cm⁻² s⁻¹")
    print(f"  Flux (FITS): {flux_fits:.3e} erg cm⁻² s⁻¹")
    print(f"  Flux ratio:  {flux_fits/flux_csv:.3f}")
    print(f"  Luminosity (CSV):  {lum_csv:.3e} erg s⁻¹")
    print(f"  Luminosity (FITS): {lum_fits:.3e} erg s⁻¹")
    print(f"  Luminosity ratio: {lum_fits/lum_csv:.3f}")
    
    # Calculate what luminosity should be
    # Using Planck18 (pipeline default)
    d_l_planck = Planck18.luminosity_distance(z).to(u.cm).value
    lum_calc_planck = 4 * np.pi * d_l_planck**2 * flux_csv * 1.0  # K=1.0
    
    # Using H0=70, Om0=0.3 (Alexis might use)
    cosmo_alexis = FlatLambdaCDM(H0=70, Om0=0.3)
    d_l_alexis = cosmo_alexis.luminosity_distance(z).to(u.cm).value
    lum_calc_alexis = 4 * np.pi * d_l_alexis**2 * flux_csv * 1.0  # K=1.0
    
    print(f"  Calculated L (Planck18, K=1.0): {lum_calc_planck:.3e} erg s⁻¹")
    print(f"  Calculated L (H0=70, K=1.0):   {lum_calc_alexis:.3e} erg s⁻¹")
    print(f"  Distance ratio (Alexis/Planck): {d_l_alexis/d_l_planck:.3f}")
    
    # Check if K-correction would help
    # For thermal emission, K(z) ≈ (1+z)^(Γ-1) for typical cases
    # But with Γ=2.0, current code gives K=1.0
    # Let's check what K should be for thermal emission
    k_simple = (1 + z) ** (2.0 - 2.0)  # Current (wrong)
    k_thermal = (1 + z) ** (1.5 - 1.0)  # Approximate for thermal
    print(f"  K-correction (current, Γ=2.0): {k_simple:.3f}")
    print(f"  K-correction (thermal, Γ=1.5): {k_thermal:.3f}")
    
    if k_thermal > 1.0:
        lum_with_k = 4 * np.pi * d_l_planck**2 * flux_csv * k_thermal
        print(f"  Calculated L (Planck18, K={k_thermal:.3f}): {lum_with_k:.3e} erg s⁻¹")
        print(f"  Ratio to FITS: {lum_fits/lum_with_k:.3f}")
    
    print()

# Check overall statistics
print("=" * 80)
print("OVERALL STATISTICS")
print("=" * 80)

# Match all sources
matched_indices = []
for i in range(len(csv_data)):
    z_csv = csv_data.iloc[i]['Redshift']
    z_diff = np.abs(fits_data['z'] - z_csv)
    if z_diff.min() < 0.01:
        matched_indices.append(i)

if len(matched_indices) > 0:
    lum_csv_all = csv_data.iloc[matched_indices]['Luminosity_erg_s'].values
    lum_fits_all = []
    for idx in matched_indices:
        z_csv = csv_data.iloc[idx]['Redshift']
        z_diff = np.abs(fits_data['z'] - z_csv)
        fits_idx = np.argmin(z_diff)
        lum_fits_all.append(fits_data.iloc[fits_idx]['Lx'])
    lum_fits_all = np.array(lum_fits_all)
    
    ratio = lum_fits_all / lum_csv_all
    ratio = ratio[ratio > 0]  # Remove invalid
    
    print(f"\nMatched {len(ratio)} sources")
    print(f"Median luminosity ratio (FITS/CSV): {np.median(ratio):.3f}")
    print(f"Mean luminosity ratio: {np.mean(ratio):.3f}")
    print(f"Std of ratio: {np.std(ratio):.3f}")
    
    # Check if it's consistent across redshift
    z_matched = csv_data.iloc[matched_indices]['Redshift'].values
    print(f"\nRedshift range: {z_matched.min():.3f} - {z_matched.max():.3f}")
    
    # Check ratio vs redshift
    for z_bin in [0.5, 1.0, 2.0, 3.0]:
        mask = (z_matched >= z_bin - 0.5) & (z_matched < z_bin + 0.5)
        if np.sum(mask) > 0:
            ratio_bin = ratio[mask]
            print(f"  z ~ {z_bin:.1f}: median ratio = {np.median(ratio_bin):.3f} (n={np.sum(mask)})")

print("\n" + "=" * 80)
print("POTENTIAL ISSUES IDENTIFIED:")
print("=" * 80)
print("1. K-correction is always 1.0 (spectral_index=2.0 gives (1+z)^0 = 1.0)")
print("2. K-correction doesn't account for energy band properly")
print("3. Cosmology difference (Planck18 vs H0=70) is small (~3%), not enough for 2.33×")
print("4. Need to check if Alexis uses proper APEC-based K-correction")
print("=" * 80)


