#!/usr/bin/env python3
"""
Comparison script for Alexis and Ghassem measurements.
Compares measurements from columns 1-18 (Alexis) with columns 20+ (Ghassem).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import astropy for cosmology, fall back to manual calculation if not available
try:
    from astropy.cosmology import FlatLambdaCDM
    from astropy import units as u
    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False
    print("Warning: astropy not available. Using simplified cosmology for R200 conversion.")

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def load_data(csv_path):
    """Load the CSV file and separate Alexis and Ghassem measurements."""
    df = pd.read_csv(csv_path)
    
    # Alexis columns (1-18, 0-indexed: 0-17)
    alexis_cols = df.columns[:18].tolist()
    alexis_data = df[alexis_cols].copy()
    
    # Catalog name (column 19, index 18)
    catalog_name = df.iloc[:, 18] if len(df.columns) > 18 else None
    
    # Ghassem columns (20+, 0-indexed: 19+)
    ghassem_cols = df.columns[19:].tolist() if len(df.columns) > 19 else []
    ghassem_data = df[ghassem_cols].copy()
    
    # Add catalog name to both if available
    if catalog_name is not None:
        alexis_data['Catalog_Name'] = catalog_name.values
        ghassem_data['Catalog_Name'] = catalog_name.values
    
    return alexis_data, ghassem_data, df

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
        # Simplified cosmology calculation (FlatLambdaCDM approximation)
        # For small z, D_A ≈ c*z / (H0 * (1+z))
        # More accurate: use numerical integration
        c = 299792.458  # km/s
        H0 = 70.0  # km/s/Mpc
        Om0 = 0.3
        Ol0 = 1.0 - Om0  # Flat universe
        
        # Angular diameter distance in Mpc (simplified)
        # D_A = D_L / (1+z)^2 where D_L is luminosity distance
        # For small z: D_L ≈ (c/H0) * z * (1 + z/2)
        z = redshift
        d_l_approx = (c / H0) * z * (1 + z / 2.0)  # Mpc, approximate
        d_a = d_l_approx / ((1 + z) ** 2)  # Mpc
        
        # Convert to kpc
        r200_kpc = r200_deg * (np.pi / 180.0) * d_a * 1000.0
    
    return r200_kpc

def compare_measurements(alexis_data, ghassem_data, full_df):
    """Compare measurements between Alexis and Ghassem."""
    
    results = {}
    
    # 1. Redshift comparison
    if 'z' in alexis_data.columns and 'Redshift' in ghassem_data.columns:
        z_alexis = alexis_data['z'].values
        z_ghassem = ghassem_data['Redshift'].values
        valid = ~(np.isnan(z_alexis) | np.isnan(z_ghassem))
        if valid.sum() > 0:
            results['redshift'] = {
                'alexis': z_alexis[valid],
                'ghassem': z_ghassem[valid],
                'name': 'Redshift',
                'unit': '',
                'alexis_col': 'z',
                'ghassem_col': 'Redshift'
            }
    
    # 2. Flux comparison (Alexis: Flux, Ghassem: Flux_erg_cm2_s)
    if 'Flux' in alexis_data.columns and 'Flux_erg_cm2_s' in ghassem_data.columns:
        flux_alexis = alexis_data['Flux'].values
        flux_ghassem = ghassem_data['Flux_erg_cm2_s'].values
        valid = ~(np.isnan(flux_alexis) | np.isnan(flux_ghassem)) & (flux_alexis > 0) & (flux_ghassem > 0)
        if valid.sum() > 0:
            results['flux'] = {
                'alexis': flux_alexis[valid],
                'ghassem': flux_ghassem[valid],
                'name': 'Flux',
                'unit': 'erg cm⁻² s⁻¹',
                'alexis_col': 'Flux',
                'ghassem_col': 'Flux_erg_cm2_s'
            }
    
    # 3. Luminosity comparison (Alexis: Lx, Ghassem: Luminosity_erg_s)
    if 'Lx' in alexis_data.columns and 'Luminosity_erg_s' in ghassem_data.columns:
        lx_alexis = alexis_data['Lx'].values
        lx_ghassem = ghassem_data['Luminosity_erg_s'].values
        valid = ~(np.isnan(lx_alexis) | np.isnan(lx_ghassem)) & (lx_alexis > 0) & (lx_ghassem > 0)
        if valid.sum() > 0:
            results['luminosity'] = {
                'alexis': lx_alexis[valid],
                'ghassem': lx_ghassem[valid],
                'name': 'Luminosity',
                'unit': 'erg s⁻¹',
                'alexis_col': 'Lx',
                'ghassem_col': 'Luminosity_erg_s'
            }
    
    # 4. Temperature comparison (Alexis: T, Ghassem: Temperature_keV)
    if 'T' in alexis_data.columns and 'Temperature_keV' in ghassem_data.columns:
        t_alexis = alexis_data['T'].values
        t_ghassem = ghassem_data['Temperature_keV'].values
        valid = ~(np.isnan(t_alexis) | np.isnan(t_ghassem)) & (t_alexis > 0) & (t_ghassem > 0)
        if valid.sum() > 0:
            results['temperature'] = {
                'alexis': t_alexis[valid],
                'ghassem': t_ghassem[valid],
                'name': 'Temperature',
                'unit': 'keV',
                'alexis_col': 'T',
                'ghassem_col': 'Temperature_keV'
            }
    
    # 5. Mass comparison (Alexis: M200, Ghassem: M200_Temp_Msun or M200_Luminosity_Msun)
    if 'M200' in alexis_data.columns:
        m200_alexis = alexis_data['M200'].values
        # Try temperature-based mass first, then luminosity-based
        if 'M200_Temp_Msun' in ghassem_data.columns:
            m200_ghassem = ghassem_data['M200_Temp_Msun'].values
            valid = ~(np.isnan(m200_alexis) | np.isnan(m200_ghassem)) & (m200_alexis > 0) & (m200_ghassem > 0)
            if valid.sum() > 0:
                results['mass_m200_temp'] = {
                    'alexis': m200_alexis[valid],
                    'ghassem': m200_ghassem[valid],
                    'name': 'M200 (Temperature-based)',
                    'unit': 'M☉',
                    'alexis_col': 'M200',
                    'ghassem_col': 'M200_Temp_Msun'
                }
        if 'M200_Luminosity_Msun' in ghassem_data.columns:
            m200_ghassem = ghassem_data['M200_Luminosity_Msun'].values
            valid = ~(np.isnan(m200_alexis) | np.isnan(m200_ghassem)) & (m200_alexis > 0) & (m200_ghassem > 0)
            if valid.sum() > 0:
                results['mass_m200_lum'] = {
                    'alexis': m200_alexis[valid],
                    'ghassem': m200_ghassem[valid],
                    'name': 'M200 (Luminosity-based)',
                    'unit': 'M☉',
                    'alexis_col': 'M200',
                    'ghassem_col': 'M200_Luminosity_Msun'
                }
    
    # 6. SNR comparison
    if 'SNR_1' in alexis_data.columns and 'SNR_2' in ghassem_data.columns:
        snr_alexis = alexis_data['SNR_1'].values
        snr_ghassem = ghassem_data['SNR_2'].values
        valid = ~(np.isnan(snr_alexis) | np.isnan(snr_ghassem)) & (snr_alexis > 0) & (snr_ghassem > 0)
        if valid.sum() > 0:
            results['snr'] = {
                'alexis': snr_alexis[valid],
                'ghassem': snr_ghassem[valid],
                'name': 'SNR',
                'unit': '',
                'alexis_col': 'SNR_1',
                'ghassem_col': 'SNR_2'
            }
    
    # 7. Position comparison (RA, DEC)
    if 'RA_1' in alexis_data.columns and 'RA_2' in ghassem_data.columns:
        ra_alexis = alexis_data['RA_1'].values
        ra_ghassem = ghassem_data['RA_2'].values
        valid = ~(np.isnan(ra_alexis) | np.isnan(ra_ghassem))
        if valid.sum() > 0:
            results['ra'] = {
                'alexis': ra_alexis[valid],
                'ghassem': ra_ghassem[valid],
                'name': 'RA',
                'unit': 'deg',
                'alexis_col': 'RA_1',
                'ghassem_col': 'RA_2'
            }
    
    if 'DEC_1' in alexis_data.columns and 'DEC_2' in ghassem_data.columns:
        dec_alexis = alexis_data['DEC_1'].values
        dec_ghassem = ghassem_data['DEC_2'].values
        valid = ~(np.isnan(dec_alexis) | np.isnan(dec_ghassem))
        if valid.sum() > 0:
            results['dec'] = {
                'alexis': dec_alexis[valid],
                'ghassem': dec_ghassem[valid],
                'name': 'DEC',
                'unit': 'deg',
                'alexis_col': 'DEC_1',
                'ghassem_col': 'DEC_2'
            }
    
    # 8. R200 comparison (Alexis: R200_deg converted to kpc, Ghassem: R200_kpc)
    if 'R200_deg' in alexis_data.columns and 'R200_kpc' in ghassem_data.columns:
        r200_deg_alexis = alexis_data['R200_deg'].values
        r200_kpc_ghassem = ghassem_data['R200_kpc'].values
        
        # Get redshift for conversion
        if 'z' in alexis_data.columns:
            z_values = alexis_data['z'].values
        elif 'Redshift' in ghassem_data.columns:
            z_values = ghassem_data['Redshift'].values
        else:
            z_values = None
        
        if z_values is not None:
            # Try to use Ghassem's luminosity distance if available for more accurate conversion
            # D_A = D_L / (1+z)^2
            if 'Luminosity_Distance_Mpc' in ghassem_data.columns:
                d_l = ghassem_data['Luminosity_Distance_Mpc'].values
                d_a = d_l / ((1 + z_values) ** 2)  # Angular diameter distance in Mpc
                r200_kpc_alexis = r200_deg_alexis * (np.pi / 180.0) * d_a * 1000.0
            else:
                # Use cosmology-based conversion
                r200_kpc_alexis = convert_r200_deg_to_kpc(r200_deg_alexis, z_values)
            
            valid = ~(np.isnan(r200_kpc_alexis) | np.isnan(r200_kpc_ghassem)) & \
                    (r200_kpc_alexis > 0) & (r200_kpc_ghassem > 0)
            
            if valid.sum() > 0:
                results['r200'] = {
                    'alexis': r200_kpc_alexis[valid],
                    'ghassem': r200_kpc_ghassem[valid],
                    'name': 'R200',
                    'unit': 'kpc',
                    'alexis_col': 'R200_deg (converted to kpc)',
                    'ghassem_col': 'R200_kpc'
                }
    
    return results

def calculate_statistics(results):
    """Calculate comparison statistics for each measurement."""
    stats_dict = {}
    
    for key, data in results.items():
        alexis = data['alexis']
        ghassem = data['ghassem']
        n = len(alexis)
        
        # Basic statistics
        diff = ghassem - alexis
        frac_diff = (ghassem - alexis) / alexis * 100  # Percentage difference
        abs_frac_diff = np.abs(frac_diff)
        
        # Correlation
        if n > 1:
            corr, p_corr = stats.pearsonr(alexis, ghassem)
            spearman, p_spearman = stats.spearmanr(alexis, ghassem)
        else:
            corr, p_corr = np.nan, np.nan
            spearman, p_spearman = np.nan, np.nan
        
        # Linear fit
        if n > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(alexis, ghassem)
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
        alexis = data['alexis']
        ghassem = data['ghassem']
        stats_data = stats_dict[key]
        
        # Scatter plot
        ax.scatter(alexis, ghassem, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        # 1:1 line
        min_val = min(np.nanmin(alexis), np.nanmin(ghassem))
        max_val = max(np.nanmax(alexis), np.nanmax(ghassem))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='1:1 line')
        
        # Best fit line
        if not np.isnan(stats_data['slope']):
            x_fit = np.linspace(min_val, max_val, 100)
            y_fit = stats_data['slope'] * x_fit + stats_data['intercept']
            ax.plot(x_fit, y_fit, 'b-', lw=2, 
                   label=f"Fit: y={stats_data['slope']:.3f}x+{stats_data['intercept']:.2e}")
        
        # Labels
        unit_str = f" ({data['unit']})" if data['unit'] else ""
        ax.set_xlabel(f"Alexis: {data['alexis_col']}{unit_str}", fontsize=11)
        ax.set_ylabel(f"Ghassem: {data['ghassem_col']}{unit_str}", fontsize=11)
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
    plt.savefig(output_dir / 'comparison_scatter_plots.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create residual plots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (key, data) in enumerate(results.items()):
        ax = axes[idx]
        alexis = data['alexis']
        ghassem = data['ghassem']
        stats_data = stats_dict[key]
        
        # Fractional difference
        frac_diff = (ghassem - alexis) / alexis * 100
        
        ax.scatter(alexis, frac_diff, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        ax.axhline(y=0, color='r', linestyle='--', lw=2)
        ax.axhline(y=stats_data['median_frac_diff'], color='b', linestyle='-', lw=2, 
                  label=f"Median: {stats_data['median_frac_diff']:.1f}%")
        ax.axhline(y=stats_data['median_frac_diff'] + stats_data['std_frac_diff'], 
                  color='b', linestyle=':', lw=1, alpha=0.7)
        ax.axhline(y=stats_data['median_frac_diff'] - stats_data['std_frac_diff'], 
                  color='b', linestyle=':', lw=1, alpha=0.7)
        
        unit_str = f" ({data['unit']})" if data['unit'] else ""
        ax.set_xlabel(f"Alexis: {data['alexis_col']}{unit_str}", fontsize=11)
        ax.set_ylabel("Fractional Difference (%)", fontsize=11)
        ax.set_title(f"{data['name']} Residuals\n"
                    f"σ={stats_data['std_frac_diff']:.1f}%, "
                    f"NMAD={stats_data['nmad']:.2e}", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        if np.nanmax(alexis) / np.nanmin(alexis) > 100:
            ax.set_xscale('log')
    
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_residual_plots.png', dpi=150, bbox_inches='tight')
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
    df_summary.to_csv(output_dir / 'comparison_summary.csv', index=False)
    
    # Also create a text summary
    with open(output_dir / 'comparison_summary.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("ALEXIS vs GHASSEM MEASUREMENT COMPARISON SUMMARY\n")
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
    csv_path = script_dir / 'alexis_ghassem.csv'
    output_dir = script_dir / 'comparison_results'
    output_dir.mkdir(exist_ok=True)
    
    print("Loading data...")
    alexis_data, ghassem_data, full_df = load_data(csv_path)
    
    print(f"Alexis columns: {len(alexis_data.columns)}")
    print(f"Ghassem columns: {len(ghassem_data.columns)}")
    print(f"Total rows: {len(full_df)}")
    
    print("\nComparing measurements...")
    results = compare_measurements(alexis_data, ghassem_data, full_df)
    
    print(f"\nFound {len(results)} comparable measurements:")
    for key, data in results.items():
        print(f"  - {data['name']}: {len(data['alexis'])} measurements")
    
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

