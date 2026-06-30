#!/usr/bin/env python3
"""
Comparison script for xray_catalog.csv (Ghassem) vs Xmass_ghassem.fits (Alexis).
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

# Try to import astropy for cosmology, fall back to manual calculation if not available
try:
    from astropy.cosmology import FlatLambdaCDM
    from astropy import units as u
    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False
    print("Warning: astropy not available. Using simplified cosmology for R200 conversion.")

def convert_r200_deg_to_kpc(r200_deg, redshift, cosmology=None):
    """
    Convert R200 from degrees to kpc using angular diameter distance.
    
    Parameters:
    -----------
    r200_deg : array-like
        R200 in degrees
    redshift : array-like
        Redshift values
    cosmology : astropy.cosmology object, optional
        Cosmology to use. If None, uses FlatLambdaCDM with H0=70, Om0=0.3
    
    Returns:
    --------
    r200_kpc : array-like
        R200 in kpc
    """
    r200_deg = np.asarray(r200_deg)
    redshift = np.asarray(redshift)
    
    if HAS_ASTROPY:
        if cosmology is None:
            # Default cosmology: H0=70 km/s/Mpc, Om0=0.3
            cosmology = FlatLambdaCDM(H0=70, Om0=0.3)
        
        # Calculate angular diameter distance
        d_a = cosmology.angular_diameter_distance(redshift)  # in Mpc
        
        # Convert: R200_kpc = R200_deg * (π/180) * D_A(Mpc) * 1000
        r200_kpc = r200_deg * (np.pi / 180.0) * d_a.value * 1000.0
    else:
        # Simplified cosmology calculation
        c = 299792.458  # km/s
        H0 = 70.0  # km/s/Mpc
        Om0 = 0.3
        z = redshift
        d_l_approx = (c / H0) * z * (1 + z / 2.0)  # Mpc, approximate
        d_a = d_l_approx / ((1 + z) ** 2)  # Mpc
        r200_kpc = r200_deg * (np.pi / 180.0) * d_a * 1000.0
    
    return r200_kpc

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
    matched_indices : tuple
        (csv_indices, fits_indices) for matched sources
    """
    csv_ra = csv_data['RA'].values
    csv_dec = csv_data['DEC'].values
    
    fits_ra = fits_data['RA'].values
    fits_dec = fits_data['DEC'].values
    
    matched_csv = []
    matched_fits = []
    
    for i, (ra_c, dec_c) in enumerate(zip(csv_ra, csv_dec)):
        # Find closest match in FITS
        ra_diff = np.abs(fits_ra - ra_c)
        dec_diff = np.abs(fits_dec - dec_c)
        
        # Combined distance
        dist = np.sqrt(ra_diff**2 + dec_diff**2)
        closest_idx = np.argmin(dist)
        
        # Check if within tolerance
        if ra_diff[closest_idx] < ra_tol and dec_diff[closest_idx] < dec_tol:
            matched_csv.append(i)
            matched_fits.append(closest_idx)
    
    return np.array(matched_csv), np.array(matched_fits)

def compare_measurements(csv_data, fits_data, csv_indices, fits_indices):
    """Compare measurements between CSV and FITS."""
    
    results = {}
    
    # 1. Redshift comparison
    if 'Redshift' in csv_data.columns and 'z' in fits_data.columns:
        z_csv = csv_data.iloc[csv_indices]['Redshift'].values
        z_fits = fits_data.iloc[fits_indices]['z'].values
        valid = ~(np.isnan(z_csv) | np.isnan(z_fits))
        if valid.sum() > 0:
            results['redshift'] = {
                'csv': z_csv[valid],
                'fits': z_fits[valid],
                'name': 'Redshift',
                'unit': '',
                'csv_col': 'Redshift',
                'fits_col': 'z'
            }
    
    # 2. Flux comparison
    if 'Flux_erg_cm2_s' in csv_data.columns and 'Flux' in fits_data.columns:
        flux_csv = csv_data.iloc[csv_indices]['Flux_erg_cm2_s'].values
        flux_fits = fits_data.iloc[fits_indices]['Flux'].values
        valid = ~(np.isnan(flux_csv) | np.isnan(flux_fits)) & (flux_csv > 0) & (flux_fits > 0)
        if valid.sum() > 0:
            results['flux'] = {
                'csv': flux_csv[valid],
                'fits': flux_fits[valid],
                'name': 'Flux',
                'unit': 'erg cm⁻² s⁻¹',
                'csv_col': 'Flux_erg_cm2_s',
                'fits_col': 'Flux'
            }
    
    # 3. Luminosity comparison
    if 'Luminosity_erg_s' in csv_data.columns and 'Lx' in fits_data.columns:
        lx_csv = csv_data.iloc[csv_indices]['Luminosity_erg_s'].values
        lx_fits = fits_data.iloc[fits_indices]['Lx'].values
        valid = ~(np.isnan(lx_csv) | np.isnan(lx_fits)) & (lx_csv > 0) & (lx_fits > 0)
        if valid.sum() > 0:
            results['luminosity'] = {
                'csv': lx_csv[valid],
                'fits': lx_fits[valid],
                'name': 'Luminosity',
                'unit': 'erg s⁻¹',
                'csv_col': 'Luminosity_erg_s',
                'fits_col': 'Lx'
            }
    
    # 4. Temperature comparison
    if 'Temperature_keV' in csv_data.columns and 'T' in fits_data.columns:
        t_csv = csv_data.iloc[csv_indices]['Temperature_keV'].values
        t_fits = fits_data.iloc[fits_indices]['T'].values
        valid = ~(np.isnan(t_csv) | np.isnan(t_fits)) & (t_csv > 0) & (t_fits > 0)
        if valid.sum() > 0:
            results['temperature'] = {
                'csv': t_csv[valid],
                'fits': t_fits[valid],
                'name': 'Temperature',
                'unit': 'keV',
                'csv_col': 'Temperature_keV',
                'fits_col': 'T'
            }
    
    # 5. Mass comparison
    if 'M200_Luminosity_Msun' in csv_data.columns and 'M200' in fits_data.columns:
        m200_csv = csv_data.iloc[csv_indices]['M200_Luminosity_Msun'].values
        m200_fits = fits_data.iloc[fits_indices]['M200'].values
        valid = ~(np.isnan(m200_csv) | np.isnan(m200_fits)) & (m200_csv > 0) & (m200_fits > 0)
        if valid.sum() > 0:
            results['mass_m200_lum'] = {
                'csv': m200_csv[valid],
                'fits': m200_fits[valid],
                'name': 'M200 (Luminosity-based)',
                'unit': 'M☉',
                'csv_col': 'M200_Luminosity_Msun',
                'fits_col': 'M200'
            }
    
    if 'M200_Temp_Msun' in csv_data.columns and 'M200' in fits_data.columns:
        m200_csv = csv_data.iloc[csv_indices]['M200_Temp_Msun'].values
        m200_fits = fits_data.iloc[fits_indices]['M200'].values
        valid = ~(np.isnan(m200_csv) | np.isnan(m200_fits)) & (m200_csv > 0) & (m200_fits > 0)
        if valid.sum() > 0:
            results['mass_m200_temp'] = {
                'csv': m200_csv[valid],
                'fits': m200_fits[valid],
                'name': 'M200 (Temperature-based)',
                'unit': 'M☉',
                'csv_col': 'M200_Temp_Msun',
                'fits_col': 'M200'
            }
    
    # 6. SNR comparison
    if 'SNR' in csv_data.columns and 'SNR' in fits_data.columns:
        snr_csv = csv_data.iloc[csv_indices]['SNR'].values
        snr_fits = fits_data.iloc[fits_indices]['SNR'].values
        valid = ~(np.isnan(snr_csv) | np.isnan(snr_fits)) & (snr_csv > 0) & (snr_fits > 0)
        if valid.sum() > 0:
            results['snr'] = {
                'csv': snr_csv[valid],
                'fits': snr_fits[valid],
                'name': 'SNR',
                'unit': '',
                'csv_col': 'SNR',
                'fits_col': 'SNR'
            }
    
    # 7. R200 comparison (convert FITS R200_deg to kpc)
    if 'R200_kpc' in csv_data.columns and 'R200_deg' in fits_data.columns:
        r200_kpc_csv = csv_data.iloc[csv_indices]['R200_kpc'].values
        
        # Get redshift for conversion
        if 'Redshift' in csv_data.columns:
            z_values = csv_data.iloc[csv_indices]['Redshift'].values
        elif 'z' in fits_data.columns:
            z_values = fits_data.iloc[fits_indices]['z'].values
        else:
            z_values = None
        
        if z_values is not None:
            r200_deg_fits = fits_data.iloc[fits_indices]['R200_deg'].values
            
            # Try to use CSV luminosity distance if available
            if 'Luminosity_Distance_Mpc' in csv_data.columns:
                d_l = csv_data.iloc[csv_indices]['Luminosity_Distance_Mpc'].values
                d_a = d_l / ((1 + z_values) ** 2)  # Angular diameter distance in Mpc
                r200_kpc_fits = r200_deg_fits * (np.pi / 180.0) * d_a * 1000.0
            else:
                r200_kpc_fits = convert_r200_deg_to_kpc(r200_deg_fits, z_values)
            
            valid = ~(np.isnan(r200_kpc_csv) | np.isnan(r200_kpc_fits)) & \
                    (r200_kpc_csv > 0) & (r200_kpc_fits > 0)
            
            if valid.sum() > 0:
                results['r200'] = {
                    'csv': r200_kpc_csv[valid],
                    'fits': r200_kpc_fits[valid],
                    'name': 'R200',
                    'unit': 'kpc',
                    'csv_col': 'R200_kpc',
                    'fits_col': 'R200_deg (converted to kpc)'
                }
    
    return results

def calculate_statistics(results):
    """Calculate comparison statistics for each measurement."""
    stats_dict = {}
    
    for key, data in results.items():
        csv_vals = data['csv']
        fits_vals = data['fits']
        n = len(csv_vals)
        
        # Basic statistics
        diff = fits_vals - csv_vals
        frac_diff = (fits_vals - csv_vals) / csv_vals * 100
        abs_frac_diff = np.abs(frac_diff)
        
        # Correlation
        if n > 1:
            corr, p_corr = stats.pearsonr(csv_vals, fits_vals)
            spearman, p_spearman = stats.spearmanr(csv_vals, fits_vals)
        else:
            corr, p_corr = np.nan, np.nan
            spearman, p_spearman = np.nan, np.nan
        
        # Linear fit
        if n > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(csv_vals, fits_vals)
        else:
            slope, intercept, r_value, p_value, std_err = np.nan, np.nan, np.nan, np.nan, np.nan
        
        # Agreement metrics
        mean_diff = np.mean(diff)
        median_diff = np.median(diff)
        std_diff = np.std(diff)
        mean_frac_diff = np.mean(frac_diff)
        median_frac_diff = np.median(frac_diff)
        std_frac_diff = np.std(frac_diff)
        mean_abs_frac_diff = np.mean(abs_frac_diff)
        median_abs_frac_diff = np.median(abs_frac_diff)
        
        # Normalized median absolute deviation (NMAD)
        nmad = 1.4826 * np.median(np.abs(diff - np.median(diff)))
        
        stats_dict[key] = {
            'n': n,
            'mean_diff': mean_diff,
            'median_diff': median_diff,
            'std_diff': std_diff,
            'mean_frac_diff': mean_frac_diff,
            'median_frac_diff': median_frac_diff,
            'std_frac_diff': std_frac_diff,
            'mean_abs_frac_diff': mean_abs_frac_diff,
            'median_abs_frac_diff': median_abs_frac_diff,
            'nmad': nmad,
            'pearson_r': corr,
            'pearson_p': p_corr,
            'spearman_r': spearman,
            'spearman_p': p_spearman,
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value**2 if not np.isnan(r_value) else np.nan,
            'p_value': p_value,
            'std_err': std_err
        }
    
    return stats_dict

def create_comparison_plots(results, stats_dict, output_dir):
    """Create comparison plots for each measurement."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    n_plots = len(results)
    if n_plots == 0:
        print("No measurements to plot!")
        return
    
    # Create a grid of subplots
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (key, data) in enumerate(results.items()):
        ax = axes[idx]
        csv_vals = data['csv']
        fits_vals = data['fits']
        stats_data = stats_dict[key]
        
        # Scatter plot
        ax.scatter(csv_vals, fits_vals, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        # 1:1 line
        min_val = min(np.nanmin(csv_vals), np.nanmin(fits_vals))
        max_val = max(np.nanmax(csv_vals), np.nanmax(fits_vals))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='1:1 line')
        
        # Best fit line
        if not np.isnan(stats_data['slope']):
            x_fit = np.linspace(min_val, max_val, 100)
            y_fit = stats_data['slope'] * x_fit + stats_data['intercept']
            ax.plot(x_fit, y_fit, 'b-', lw=2, 
                   label=f"Fit: y={stats_data['slope']:.3f}x+{stats_data['intercept']:.2e}")
        
        # Labels
        unit_str = f" ({data['unit']})" if data['unit'] else ""
        ax.set_xlabel(f"CSV (Ghassem): {data['csv_col']}{unit_str}", fontsize=11)
        ax.set_ylabel(f"FITS (Alexis): {data['fits_col']}{unit_str}", fontsize=11)
        ax.set_title(f"{data['name']} Comparison\n"
                    f"r={stats_data['pearson_r']:.3f}, "
                    f"n={stats_data['n']}, "
                    f"Δ={stats_data['median_frac_diff']:.1f}%", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Log scale if values span orders of magnitude
        if max_val / min_val > 100:
            ax.set_xscale('log')
            ax.set_yscale('log')
    
    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'csv_fits_comparison_scatter_plots.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create residual plots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (key, data) in enumerate(results.items()):
        ax = axes[idx]
        csv_vals = data['csv']
        fits_vals = data['fits']
        stats_data = stats_dict[key]
        
        # Fractional difference
        frac_diff = (fits_vals - csv_vals) / csv_vals * 100
        
        ax.scatter(csv_vals, frac_diff, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        ax.axhline(y=0, color='r', linestyle='--', lw=2)
        ax.axhline(y=stats_data['median_frac_diff'], color='b', linestyle='-', lw=2, 
                  label=f"Median: {stats_data['median_frac_diff']:.1f}%")
        ax.axhline(y=stats_data['median_frac_diff'] + stats_data['std_frac_diff'], 
                  color='b', linestyle=':', lw=1, alpha=0.7)
        ax.axhline(y=stats_data['median_frac_diff'] - stats_data['std_frac_diff'], 
                  color='b', linestyle=':', lw=1, alpha=0.7)
        
        unit_str = f" ({data['unit']})" if data['unit'] else ""
        ax.set_xlabel(f"CSV (Ghassem): {data['csv_col']}{unit_str}", fontsize=11)
        ax.set_ylabel("Fractional Difference (%)", fontsize=11)
        ax.set_title(f"{data['name']} Residuals\n"
                    f"σ={stats_data['std_frac_diff']:.1f}%, "
                    f"NMAD={stats_data['nmad']:.2e}", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        if np.nanmax(csv_vals) / np.nanmin(csv_vals) > 100:
            ax.set_xscale('log')
    
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'csv_fits_comparison_residual_plots.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_summary_table(stats_dict, results, output_dir):
    """Create a summary statistics table."""
    output_dir = Path(output_dir)
    
    summary_data = []
    for key, stats_data in stats_dict.items():
        data = results[key]
        summary_data.append({
            'Measurement': data['name'],
            'N': stats_data['n'],
            'Pearson r': f"{stats_data['pearson_r']:.4f}",
            'Pearson p': f"{stats_data['pearson_p']:.4e}" if not np.isnan(stats_data['pearson_p']) else 'N/A',
            'Spearman ρ': f"{stats_data['spearman_r']:.4f}",
            'Slope': f"{stats_data['slope']:.4f}",
            'R²': f"{stats_data['r_squared']:.4f}",
            'Median Δ (%)': f"{stats_data['median_frac_diff']:.2f}",
            'Mean |Δ| (%)': f"{stats_data['mean_abs_frac_diff']:.2f}",
            'Std Δ (%)': f"{stats_data['std_frac_diff']:.2f}",
            'NMAD': f"{stats_data['nmad']:.2e}"
        })
    
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(output_dir / 'csv_fits_comparison_summary.csv', index=False)
    
    # Also create a text summary
    with open(output_dir / 'csv_fits_comparison_summary.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CSV (Ghassem) vs FITS (Alexis) MEASUREMENT COMPARISON SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        for key, stats_data in stats_dict.items():
            data = results[key]
            f.write(f"\n{data['name']} ({data['unit']})\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Number of measurements: {stats_data['n']}\n")
            f.write(f"  Pearson correlation: r = {stats_data['pearson_r']:.4f}, p = {stats_data['pearson_p']:.4e}\n")
            f.write(f"  Spearman correlation: ρ = {stats_data['spearman_r']:.4f}, p = {stats_data['spearman_p']:.4e}\n")
            f.write(f"  Linear fit: y = {stats_data['slope']:.4f}x + {stats_data['intercept']:.2e}\n")
            f.write(f"  R² = {stats_data['r_squared']:.4f}\n")
            f.write(f"  Median difference: {stats_data['median_diff']:.2e} ({stats_data['median_frac_diff']:.2f}%)\n")
            f.write(f"  Mean absolute difference: {stats_data['mean_abs_frac_diff']:.2f}%\n")
            f.write(f"  Std of differences: {stats_data['std_frac_diff']:.2f}%\n")
            f.write(f"  NMAD: {stats_data['nmad']:.2e}\n")
    
    return df_summary

def main():
    """Main function."""
    # Paths
    script_dir = Path(__file__).parent
    csv_path = script_dir / 'xray_catalog.csv'
    fits_path = script_dir / 'Xmass_ghassem.fits'
    output_dir = script_dir / 'csv_fits_comparison_results'
    output_dir.mkdir(exist_ok=True)
    
    print("Loading data...")
    csv_data, fits_data = load_data(csv_path, fits_path)
    
    print(f"CSV rows: {len(csv_data)}")
    print(f"FITS rows: {len(fits_data)}")
    print(f"CSV columns: {list(csv_data.columns[:10])}...")
    print(f"FITS columns: {list(fits_data.columns)}")
    
    print("\nMatching sources by RA/DEC...")
    csv_indices, fits_indices = match_sources(csv_data, fits_data)
    print(f"Matched {len(csv_indices)} sources")
    
    if len(csv_indices) == 0:
        print("ERROR: No sources matched! Check RA/DEC columns and coordinate systems.")
        return
    
    print("\nComparing measurements...")
    results = compare_measurements(csv_data, fits_data, csv_indices, fits_indices)
    
    print(f"\nFound {len(results)} comparable measurements:")
    for key, data in results.items():
        print(f"  - {data['name']}: {len(data['csv'])} measurements")
    
    print("\nCalculating statistics...")
    stats_dict = calculate_statistics(results)
    
    print("\nCreating plots...")
    create_comparison_plots(results, stats_dict, output_dir)
    
    print("\nCreating summary table...")
    summary_df = create_summary_table(stats_dict, results, output_dir)
    
    print(f"\nResults saved to: {output_dir}")
    print("\nSummary Statistics:")
    print(summary_df.to_string(index=False))
    
    return results, stats_dict, summary_df

if __name__ == '__main__':
    results, stats_dict, summary_df = main()


