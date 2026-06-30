#!/usr/bin/env python3
"""
Comparison script for detections.csv (Ghassem) vs Xmass_ghassem.fits (Alexis).
Compares measurements from the same data sources.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from astropy.io import fits
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def load_data(csv_path, fits_path):
    """Load CSV and FITS files."""
    # Load CSV (Ghassem)
    csv_data = pd.read_csv(csv_path)
    
    # Load FITS (Alexis)
    with fits.open(fits_path) as hdul:
        # Get the data extension (usually extension 1)
        hdu = hdul[1] if len(hdul) > 1 else hdul[0]
        fits_data = hdu.data
        fits_df = pd.DataFrame(fits_data)
    
    return csv_data, fits_df

def match_sources(csv_data, fits_data, ra_tol=0.001, dec_tol=0.001):
    """
    Match sources between CSV and FITS by RA/DEC.
    
    Parameters:
    -----------
    csv_data : pd.DataFrame
        CSV data (Ghassem)
    fits_data : pd.DataFrame
        FITS data (Alexis)
    ra_tol : float
        RA matching tolerance in degrees
    dec_tol : float
        DEC matching tolerance in degrees
    
    Returns:
    --------
    matched_indices : dict
        Dictionary with 'csv' and 'fits' keys containing matched indices
    """
    matched_csv = []
    matched_fits = []
    
    for i, csv_row in csv_data.iterrows():
        csv_ra = csv_row['RA']
        csv_dec = csv_row['DEC']
        
        # Find closest match in FITS
        ra_diff = np.abs(fits_data['RA'] - csv_ra)
        dec_diff = np.abs(fits_data['DEC'] - csv_dec)
        
        # Match if within tolerance
        match_mask = (ra_diff < ra_tol) & (dec_diff < dec_tol)
        
        if match_mask.sum() > 0:
            # Take the closest match
            dist = np.sqrt(ra_diff**2 + dec_diff**2)
            best_match = np.argmin(dist[match_mask])
            fits_idx = np.where(match_mask)[0][best_match]
            
            matched_csv.append(i)
            matched_fits.append(fits_idx)
    
    return {'csv': matched_csv, 'fits': matched_fits}

def compare_measurements(csv_data, fits_data, matched_indices):
    """Compare measurements between CSV and FITS."""
    results = {}
    
    csv_matched = csv_data.iloc[matched_indices['csv']].reset_index(drop=True)
    fits_matched = fits_data.iloc[matched_indices['fits']].reset_index(drop=True)
    
    # 1. Redshift
    if 'Redshift' in csv_matched.columns and 'z' in fits_matched.columns:
        results['redshift'] = {
            'csv': csv_matched['Redshift'].values,
            'fits': fits_matched['z'].values,
            'name': 'Redshift',
            'unit': ''
        }
    
    # 2. Flux
    if 'Flux_erg_cm2_s' in csv_matched.columns and 'Flux' in fits_matched.columns:
        results['flux'] = {
            'csv': csv_matched['Flux_erg_cm2_s'].values,
            'fits': fits_matched['Flux'].values,
            'name': 'Flux',
            'unit': 'erg cm⁻² s⁻¹'
        }
    
    # 3. Luminosity
    if 'Luminosity_erg_s' in csv_matched.columns and 'Lx' in fits_matched.columns:
        results['luminosity'] = {
            'csv': csv_matched['Luminosity_erg_s'].values,
            'fits': fits_matched['Lx'].values,
            'name': 'Luminosity',
            'unit': 'erg s⁻¹'
        }
    
    # 4. Temperature
    if 'Temperature_keV' in csv_matched.columns and 'T' in fits_matched.columns:
        results['temperature'] = {
            'csv': csv_matched['Temperature_keV'].values,
            'fits': fits_matched['T'].values,
            'name': 'Temperature',
            'unit': 'keV'
        }
    
    # 5. M200 (Luminosity-based)
    if 'M200_Luminosity_Msun' in csv_matched.columns and 'M200' in fits_matched.columns:
        results['m200_lum'] = {
            'csv': csv_matched['M200_Luminosity_Msun'].values,
            'fits': fits_matched['M200'].values,
            'name': 'M200 (Luminosity-based)',
            'unit': 'M☉'
        }
    
    # 6. M200 (Temperature-based)
    if 'M200_Temp_Msun' in csv_matched.columns and 'M200' in fits_matched.columns:
        results['m200_temp'] = {
            'csv': csv_matched['M200_Temp_Msun'].values,
            'fits': fits_matched['M200'].values,
            'name': 'M200 (Temperature-based)',
            'unit': 'M☉'
        }
    
    # 7. SNR
    if 'SNR' in csv_matched.columns and 'SNR' in fits_matched.columns:
        results['snr'] = {
            'csv': csv_matched['SNR'].values,
            'fits': fits_matched['SNR'].values,
            'name': 'SNR',
            'unit': ''
        }
    
    # 8. R200 (Luminosity-based)
    if 'R200_Luminosity_kpc' in csv_matched.columns and 'R200_deg' in fits_matched.columns:
        # Convert FITS R200 from degrees to kpc
        from astropy.cosmology import FlatLambdaCDM
        cosmo = FlatLambdaCDM(H0=67.4, Om0=0.315)
        z_vals = fits_matched['z'].values
        d_a = cosmo.angular_diameter_distance(z_vals).value  # Mpc
        r200_fits_kpc = fits_matched['R200_deg'].values * (np.pi / 180.0) * d_a * 1000.0
        
        results['r200_lum'] = {
            'csv': csv_matched['R200_Luminosity_kpc'].values,
            'fits': r200_fits_kpc,
            'name': 'R200 (Luminosity-based)',
            'unit': 'kpc'
        }
    
    # 9. R200 (Temperature-based)
    if 'R200_Temp_kpc' in csv_matched.columns and 'R200_deg' in fits_matched.columns:
        # Convert FITS R200 from degrees to kpc
        from astropy.cosmology import FlatLambdaCDM
        cosmo = FlatLambdaCDM(H0=67.4, Om0=0.315)
        z_vals = fits_matched['z'].values
        d_a = cosmo.angular_diameter_distance(z_vals).value  # Mpc
        r200_fits_kpc = fits_matched['R200_deg'].values * (np.pi / 180.0) * d_a * 1000.0
        
        results['r200_temp'] = {
            'csv': csv_matched['R200_Temp_kpc'].values,
            'fits': r200_fits_kpc,
            'name': 'R200 (Temperature-based)',
            'unit': 'kpc'
        }
    
    return results

def calculate_statistics(results):
    """Calculate comparison statistics for each measurement."""
    stats_list = []
    
    for key, data in results.items():
        csv_vals = data['csv']
        fits_vals = data['fits']
        
        # Filter out invalid values
        valid = ~(np.isnan(csv_vals) | np.isnan(fits_vals)) & \
                (csv_vals > 0) & (fits_vals > 0) & \
                np.isfinite(csv_vals) & np.isfinite(fits_vals)
        
        if valid.sum() < 3:
            continue
        
        csv_valid = csv_vals[valid]
        fits_valid = fits_vals[valid]
        
        # Correlations
        pearson_r, pearson_p = stats.pearsonr(csv_valid, fits_valid)
        spearman_rho, spearman_p = stats.spearmanr(csv_valid, fits_valid)
        
        # Linear fit
        slope, intercept, r_value, p_value, std_err = stats.linregress(csv_valid, fits_valid)
        r_squared = r_value**2
        
        # Differences
        diff = fits_valid - csv_valid
        diff_pct = 100 * diff / csv_valid
        
        median_diff = np.median(diff)
        median_diff_pct = np.median(diff_pct)
        mean_abs_diff_pct = np.mean(np.abs(diff_pct))
        std_diff_pct = np.std(diff_pct)
        
        # NMAD (Normalized Median Absolute Deviation)
        nmad = 1.4826 * np.median(np.abs(diff - np.median(diff)))
        
        stats_list.append({
            'Measurement': data['name'],
            'N': len(csv_valid),
            'Pearson r': pearson_r,
            'Pearson p': pearson_p,
            'Spearman ρ': spearman_rho,
            'Slope': slope,
            'R²': r_squared,
            'Median Δ (%)': median_diff_pct,
            'Mean |Δ| (%)': mean_abs_diff_pct,
            'Std Δ (%)': std_diff_pct,
            'NMAD': nmad
        })
    
    return pd.DataFrame(stats_list)

def create_plots(results, output_dir):
    """Create comparison plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_measurements = len(results)
    n_cols = 3
    n_rows = (n_measurements + n_cols - 1) // n_cols
    
    # Scatter plots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_measurements > 1 else [axes]
    
    for idx, (key, data) in enumerate(results.items()):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        csv_vals = data['csv']
        fits_vals = data['fits']
        
        # Filter valid values
        valid = ~(np.isnan(csv_vals) | np.isnan(fits_vals)) & \
                (csv_vals > 0) & (fits_vals > 0) & \
                np.isfinite(csv_vals) & np.isfinite(fits_vals)
        
        if valid.sum() < 3:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(data['name'])
            continue
        
        csv_valid = csv_vals[valid]
        fits_valid = fits_vals[valid]
        
        # Scatter plot
        ax.scatter(csv_valid, fits_valid, alpha=0.6, s=30)
        
        # 1:1 line
        min_val = min(np.min(csv_valid), np.min(fits_valid))
        max_val = max(np.max(csv_valid), np.max(fits_valid))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='1:1')
        
        # Fit line
        slope, intercept, r_value, _, _ = stats.linregress(csv_valid, fits_valid)
        fit_line = slope * np.array([min_val, max_val]) + intercept
        ax.plot([min_val, max_val], fit_line, 'b-', alpha=0.7, 
                label=f'y={slope:.3f}x+{intercept:.2e}\nR²={r_value**2:.3f}')
        
        ax.set_xlabel(f"CSV ({data['unit']})")
        ax.set_ylabel(f"FITS ({data['unit']})")
        ax.set_title(data['name'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_yscale('log')
    
    # Hide unused subplots
    for idx in range(len(results), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'detections_fits_comparison_scatter_plots.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Residual plots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_measurements > 1 else [axes]
    
    for idx, (key, data) in enumerate(results.items()):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        csv_vals = data['csv']
        fits_vals = data['fits']
        
        # Filter valid values
        valid = ~(np.isnan(csv_vals) | np.isnan(fits_vals)) & \
                (csv_vals > 0) & (fits_vals > 0) & \
                np.isfinite(csv_vals) & np.isfinite(fits_vals)
        
        if valid.sum() < 3:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(data['name'])
            continue
        
        csv_valid = csv_vals[valid]
        fits_valid = fits_vals[valid]
        
        # Residuals
        residuals = (fits_valid - csv_valid) / csv_valid * 100  # Percentage
        
        ax.scatter(csv_valid, residuals, alpha=0.6, s=30)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.axhline(y=np.median(residuals), color='b', linestyle='-', alpha=0.7, 
                   label=f'Median: {np.median(residuals):.1f}%')
        
        ax.set_xlabel(f"CSV ({data['unit']})")
        ax.set_ylabel('Residual (%)')
        ax.set_title(f"{data['name']} - Residuals")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
    
    # Hide unused subplots
    for idx in range(len(results), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'detections_fits_comparison_residual_plots.png', dpi=150, bbox_inches='tight')
    plt.close()

def main():
    """Main comparison function."""
    csv_path = '/Users/gozalig1/Projects/compact-groups-xray-analysis/outputs/results_snr2p0/cw_hcg/detections.csv'
    fits_path = '/Users/gozalig1/Projects/compact-groups-xray-analysis/alexis_ghassem_measurments/Xmass_ghassem.fits'
    output_dir = '/Users/gozalig1/Projects/compact-groups-xray-analysis/alexis_ghassem_measurments/detections_fits_comparison_results'
    
    print("Loading data...")
    csv_data, fits_data = load_data(csv_path, fits_path)
    print(f"CSV rows: {len(csv_data)}")
    print(f"FITS rows: {len(fits_data)}")
    print(f"CSV columns: {list(csv_data.columns)[:10]}...")
    print(f"FITS columns: {list(fits_data.columns)}")
    print()
    
    print("Matching sources by RA/DEC...")
    matched_indices = match_sources(csv_data, fits_data)
    print(f"Matched {len(matched_indices['csv'])} sources")
    print()
    
    print("Comparing measurements...")
    results = compare_measurements(csv_data, fits_data, matched_indices)
    print()
    print(f"Found {len(results)} comparable measurements:")
    for key, data in results.items():
        n_valid = np.sum(~(np.isnan(data['csv']) | np.isnan(data['fits'])) & 
                        (data['csv'] > 0) & (data['fits'] > 0))
        print(f"  - {data['name']}: {n_valid} measurements")
    print()
    
    print("Calculating statistics...")
    stats_df = calculate_statistics(results)
    print()
    
    print("Creating plots...")
    create_plots(results, output_dir)
    print()
    
    print("Creating summary table...")
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Save statistics
    stats_df.to_csv(output_dir_path / 'detections_fits_comparison_summary.csv', index=False)
    
    # Create text summary
    with open(output_dir_path / 'detections_fits_comparison_summary.txt', 'w') as f:
        f.write("Detections.csv vs FITS Comparison Summary\n")
        f.write("=" * 80 + "\n\n")
        
        for _, row in stats_df.iterrows():
            f.write(f"{row['Measurement']} ({row['N']} measurements)\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Pearson correlation: r = {row['Pearson r']:.4f}, p = {row['Pearson p']:.2e}\n")
            f.write(f"  Spearman correlation: ρ = {row['Spearman ρ']:.4f}\n")
            f.write(f"  Linear fit: y = {row['Slope']:.4f}x + ...\n")
            f.write(f"  R² = {row['R²']:.4f}\n")
            f.write(f"  Median difference: {row['Median Δ (%)']:.2f}%\n")
            f.write(f"  Mean absolute difference: {row['Mean |Δ| (%)']:.2f}%\n")
            f.write(f"  Std of differences: {row['Std Δ (%)']:.2f}%\n")
            f.write(f"  NMAD: {row['NMAD']:.2e}\n")
            f.write("\n")
    
    print(f"Results saved to: {output_dir}")
    print()
    print("Summary Statistics:")
    print(stats_df.to_string(index=False))

if __name__ == '__main__':
    main()


