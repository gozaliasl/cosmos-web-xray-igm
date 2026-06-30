#!/usr/bin/env python
"""
Create catalog representation plots including redshift distribution.

Plots:
1. Redshift distribution (histogram + cumulative overlaid)
2. Redshift vs richness (if available)
3. Sky distribution colored by redshift
4. Detection rate vs redshift bins
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from pathlib import Path
import matplotlib

matplotlib.use('Agg')  # Non-interactive backend

# Paths
BASE_DIR = Path('/Users/gozalig1/Projects/compact-groups-xray-analysis')
OUTPUT_DIR = BASE_DIR / 'cosmos-web_galaxy-groups-X-ray-properties' / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Catalog paths
CW_ALL_CATALOG = BASE_DIR / 'data' / 'group-catalog' / 'cosmos_web_groups_catalog_refined_z.fits'
CW_HCG_CATALOG = BASE_DIR / 'data' / 'group-catalog' / 'Py18_Groups_refined_z.fits'

# X-ray results paths
CW_ALL_XRAY = BASE_DIR / 'outputs' / 'results' / 'cw_all' / 'xray_catalog.fits'
CW_HCG_XRAY = BASE_DIR / 'outputs' / 'results' / 'cw_hcg' / 'xray_catalog.fits'


def load_catalog(catalog_path):
    """Load catalog and extract redshifts."""
    table = Table.read(catalog_path)
    
    # Find redshift column
    z_col = None
    for col in ['z', 'Redshift', 'Z', 'Group_z', 'z_vrf_refined']:
        if col in table.colnames:
            z_col = col
            break
    
    if z_col is None:
        raise ValueError(f"No redshift column found in {catalog_path}")
    
    redshifts = np.array(table[z_col])
    
    # Get other useful columns if available
    data = {}
    data['z'] = redshifts
    
    # Try to get RA, Dec
    for ra_col in ['RA', 'Ra', 'ra']:
        if ra_col in table.colnames:
            data['ra'] = np.array(table[ra_col])
            break
    
    for dec_col in ['DEC', 'Dec', 'dec']:
        if dec_col in table.colnames:
            data['dec'] = np.array(table[dec_col])
            break
    
    # Try to get richness
    for rich_col in ['LAMBDA_STAR', 'lambda_star', 'Richness', 'richness']:
        if rich_col in table.colnames:
            data['richness'] = np.array(table[rich_col])
            break
    
    return data, table


def load_xray_results(xray_path):
    """Load X-ray analysis results."""
    if not xray_path.exists():
        return None
    
    table = Table.read(xray_path)
    
    # Find detection column
    det_col = None
    for col in ['Is_Detected', 'detected', 'DETECTED', 'SNR']:
        if col in table.colnames:
            det_col = col
            break
    
    if det_col is None:
        return None
    
    # Get detection flags
    if det_col == 'SNR':
        detected = np.array(table[det_col]) >= 2.0
    else:
        detected = np.array(table[det_col], dtype=bool)
    
    # Get redshifts
    z_col = None
    for col in ['Redshift', 'z', 'Z']:
        if col in table.colnames:
            z_col = col
            break
    
    if z_col is None:
        return None
    
    redshifts = np.array(table[z_col])
    
    return {'detected': detected, 'z': redshifts}


def plot_redshift_distribution(cw_all_data, cw_hcg_data, cw_all_xray=None, cw_hcg_xray=None):
    """Create redshift distribution plot."""
    fig, axes = plt.subplots(2, 1, figsize=(6, 8))
    
    # Panel 1: Histogram with overlaid cumulative distribution
    ax = axes[0]
    bins = np.linspace(0, 4, 41)
    
    # Histogram
    hist_all = ax.hist(cw_all_data['z'], bins=bins, alpha=0.6, label='CW-All (N=1678)', 
            color='steelblue', edgecolor='black', linewidth=0.5)
    hist_hcg = ax.hist(cw_hcg_data['z'], bins=bins, alpha=0.6, label='CW-HCG (N=912)', 
            color='coral', edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Redshift $z$', fontsize=12)
    ax.set_ylabel('Number of Groups', fontsize=12)
    ax.set_title('(a) Redshift Distribution', fontsize=13, fontweight='bold')
    ax.tick_params(axis='both', labelsize=11)
    ax.set_xlim(0, 4)
    
    # Add cumulative distribution on secondary y-axis (right side)
    ax2 = ax.twinx()
    z_all_sorted = np.sort(cw_all_data['z'])
    z_hcg_sorted = np.sort(cw_hcg_data['z'])
    
    z_plot = np.linspace(0, 4, 1000)
    cum_all = np.array([np.sum(z_all_sorted <= z) for z in z_plot]) / len(z_all_sorted)
    cum_hcg = np.array([np.sum(z_hcg_sorted <= z) for z in z_plot]) / len(z_hcg_sorted)
    
    line_all = ax2.plot(z_plot, cum_all * 100, label='CW-All (cumulative)', color='steelblue', 
             linewidth=2, linestyle='--', alpha=0.8)
    line_hcg = ax2.plot(z_plot, cum_hcg * 100, label='CW-HCG (cumulative)', color='coral', 
             linewidth=2, linestyle='--', alpha=0.8)
    
    ax2.set_ylabel('Cumulative Fraction (\%)', fontsize=12, color='black')
    ax2.tick_params(axis='y', labelcolor='black', labelsize=11)
    ax2.set_ylim(0, 100)
    
    # Store legend elements for panel (a) only
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D
    
    # Create custom legend elements
    hist_patch_all = Rectangle((0, 0), 1, 1, facecolor='steelblue', alpha=0.6, edgecolor='black', linewidth=0.5)
    hist_patch_hcg = Rectangle((0, 0), 1, 1, facecolor='coral', alpha=0.6, edgecolor='black', linewidth=0.5)
    line_patch_all = Line2D([0], [0], color='steelblue', linewidth=2, linestyle='--', alpha=0.8)
    line_patch_hcg = Line2D([0], [0], color='coral', linewidth=2, linestyle='--', alpha=0.8)
    
    all_legend_elements = [hist_patch_all, hist_patch_hcg, line_patch_all, line_patch_hcg]
    all_legend_labels = ['CW-All (N=1678)', 'CW-HCG (N=912)', 'CW-All (cumulative)', 'CW-HCG (cumulative)']
    
    # Add legend to panel (a) on the right side, centered vertically
    ax.legend(all_legend_elements, all_legend_labels, loc='center right', fontsize=10, 
              frameon=True, fancybox=False, shadow=False)
    
    # Panel 2: Detection rate vs redshift
    ax = axes[1]
    
    if cw_all_xray is not None and cw_hcg_xray is not None:
        z_bins = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0])
        z_centers = (z_bins[:-1] + z_bins[1:]) / 2
        
        # CW-All detection rates
        det_rates_all = []
        for i in range(len(z_bins) - 1):
            mask = (cw_all_xray['z'] >= z_bins[i]) & (cw_all_xray['z'] < z_bins[i+1])
            if mask.sum() > 0:
                det_rate = cw_all_xray['detected'][mask].sum() / mask.sum() * 100
            else:
                det_rate = 0
            det_rates_all.append(det_rate)
        
        # CW-HCG detection rates
        det_rates_hcg = []
        for i in range(len(z_bins) - 1):
            mask = (cw_hcg_xray['z'] >= z_bins[i]) & (cw_hcg_xray['z'] < z_bins[i+1])
            if mask.sum() > 0:
                det_rate = cw_hcg_xray['detected'][mask].sum() / mask.sum() * 100
            else:
                det_rate = 0
            det_rates_hcg.append(det_rate)
        
        width = 0.35
        x_all = z_centers - width/2
        x_hcg = z_centers + width/2
        
        ax.bar(x_all, det_rates_all, width, label='CW-All', color='steelblue', alpha=0.7, edgecolor='black')
        ax.bar(x_hcg, det_rates_hcg, width, label='CW-HCG', color='coral', alpha=0.7, edgecolor='black')
        
        ax.set_xlabel('Redshift $z$', fontsize=12)
        ax.set_ylabel('Detection Rate (\%)', fontsize=12)
        ax.set_title('(b) X-ray Detection Rate vs Redshift', fontsize=13, fontweight='bold')
        ax.tick_params(axis='both', labelsize=11)
        ax.set_xlim(0, 3.5)
        ax.set_ylim(0, 100)  # Show full range up to 100%
    else:
        ax.text(0.5, 0.5, 'X-ray detection data\nnot available', 
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.set_title('(b) X-ray Detection Rate vs Redshift', fontsize=12, fontweight='bold')
    
    # No legend - removed as requested
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'redshift_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_redshift_richness(cw_all_data, cw_hcg_data):
    """Plot redshift vs richness if richness data available."""
    if 'richness' not in cw_all_data or 'richness' not in cw_hcg_data:
        print("Richness data not available, skipping richness plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter out invalid richness values
    mask_all = (cw_all_data['richness'] > 0) & np.isfinite(cw_all_data['richness'])
    mask_hcg = (cw_hcg_data['richness'] > 0) & np.isfinite(cw_hcg_data['richness'])
    
    ax.scatter(cw_all_data['z'][mask_all], cw_all_data['richness'][mask_all],
               alpha=0.4, s=20, label='CW-All', color='steelblue', edgecolors='none')
    ax.scatter(cw_hcg_data['z'][mask_hcg], cw_hcg_data['richness'][mask_hcg],
               alpha=0.4, s=20, label='CW-HCG', color='coral', edgecolors='none')
    
    ax.set_xlabel('Redshift $z$', fontsize=12)
    ax.set_ylabel('Richness $\lambda_{\star}$', fontsize=12)
    ax.set_title('Redshift vs Richness', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_xlim(0, 4)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'redshift_richness.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    """Main function."""
    print("Loading catalogs...")
    
    # Load catalogs
    cw_all_data, cw_all_table = load_catalog(CW_ALL_CATALOG)
    cw_hcg_data, cw_hcg_table = load_catalog(CW_HCG_CATALOG)
    
    print(f"CW-All: {len(cw_all_data['z'])} groups, z = [{np.min(cw_all_data['z']):.2f}, {np.max(cw_all_data['z']):.2f}]")
    print(f"CW-HCG: {len(cw_hcg_data['z'])} groups, z = [{np.min(cw_hcg_data['z']):.2f}, {np.max(cw_hcg_data['z']):.2f}]")
    
    # Load X-ray results if available
    cw_all_xray = load_xray_results(CW_ALL_XRAY)
    cw_hcg_xray = load_xray_results(CW_HCG_XRAY)
    
    if cw_all_xray is not None:
        print(f"CW-All X-ray: {cw_all_xray['detected'].sum()}/{len(cw_all_xray['detected'])} detected")
    if cw_hcg_xray is not None:
        print(f"CW-HCG X-ray: {cw_hcg_xray['detected'].sum()}/{len(cw_hcg_xray['detected'])} detected")
    
    # Create plots
    print("\nCreating redshift distribution plot...")
    plot_redshift_distribution(cw_all_data, cw_hcg_data, cw_all_xray, cw_hcg_xray)
    
    print("\nCreating redshift vs richness plot...")
    plot_redshift_richness(cw_all_data, cw_hcg_data)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
