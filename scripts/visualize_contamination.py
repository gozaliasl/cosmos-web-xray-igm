#!/usr/bin/env python3
"""
Visualize projected contamination from low-z groups affecting high-z groups.

Creates plots showing:
1. Spatial distribution of contaminated groups
2. Contamination severity vs redshift
3. Contaminant properties
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.table import Table
import argparse
import logging
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_catalog(catalog_slug: str, results_dir: Path) -> Optional[Table]:
    """Load catalog results."""
    catalog_path = results_dir / catalog_slug / "xray_catalog.fits"
    if not catalog_path.exists():
        catalog_path = results_dir / catalog_slug / "xray_catalog.csv"
        if not catalog_path.exists():
            logger.warning(f"No catalog data found for {catalog_slug}")
            return None
    
    logger.info(f"Loading {catalog_path.name}")
    return Table.read(catalog_path)


def plot_contamination_analysis(
    cw_all_table: Optional[Table],
    cw_hcg_table: Optional[Table],
    output_dir: Path,
    figsize: tuple = (14, 10),
    dpi: int = 150,
    show: bool = False
):
    """
    Create contamination analysis plots.
    
    Parameters
    ----------
    cw_all_table : Table or None
        Catalog table for CW-All
    cw_hcg_table : Table or None
        Catalog table for CW-HCG
    output_dir : Path
        Output directory for plots
    figsize : tuple
        Figure size
    dpi : int
        Figure resolution
    show : bool
        Whether to display plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if 'Is_Projected_Contaminated' not in (cw_all_table.colnames if cw_all_table is not None else []) and \
       'Is_Projected_Contaminated' not in (cw_hcg_table.colnames if cw_hcg_table is not None else []):
        logger.warning("No contamination data found. Run analysis with check_projected_contamination: true")
        return
    
    # Create multi-panel figure
    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi)
    
    colors = {'cw_all': '#1f77b4', 'cw_hcg': '#ff7f0e'}
    markers = {'cw_all': 'o', 'cw_hcg': 's'}
    labels = {'cw_all': 'CW-All', 'cw_hcg': 'CW-HCG'}
    
    all_redshift = []
    all_contaminated = []
    all_severity = []
    all_m200 = []
    
    for catalog_slug, table in [('cw_all', cw_all_table), ('cw_hcg', cw_hcg_table)]:
        if table is None or 'Is_Projected_Contaminated' not in table.colnames:
            continue
        
        redshift = np.array(table['Redshift'], dtype=float)
        contaminated = np.array(table['Is_Projected_Contaminated'], dtype=bool)
        severity = np.array(table['Contamination_Severity'], dtype=float)
        m200 = np.array(table['M200_Luminosity_Msun'], dtype=float) if 'M200_Luminosity_Msun' in table.colnames else None
        
        # Filter valid data
        valid = np.isfinite(redshift) & (redshift > 0)
        if m200 is not None:
            valid &= np.isfinite(m200) & (m200 > 0)
        
        z_valid = redshift[valid]
        contam_valid = contaminated[valid]
        sev_valid = severity[valid]
        m200_valid = m200[valid] if m200 is not None else None
        
        all_redshift.extend(z_valid)
        all_contaminated.extend(contam_valid)
        all_severity.extend(sev_valid)
        if m200_valid is not None:
            all_m200.extend(m200_valid)
        
        # Plot 1: Contamination fraction vs redshift
        z_bins = np.linspace(0.5, 3.5, 15)
        z_centers = (z_bins[:-1] + z_bins[1:]) / 2
        contamination_fraction = []
        
        for i in range(len(z_bins) - 1):
            mask = (z_valid >= z_bins[i]) & (z_valid < z_bins[i+1])
            if np.sum(mask) > 0:
                frac = np.sum(contam_valid[mask]) / np.sum(mask)
            else:
                frac = np.nan
            contamination_fraction.append(frac)
        
        axes[0, 0].plot(z_centers, contamination_fraction, marker=markers[catalog_slug], 
                        color=colors[catalog_slug], label=labels[catalog_slug], linewidth=2, markersize=6)
        
        # Plot 2: Contamination severity vs redshift (scatter)
        if np.any(contam_valid):
            axes[0, 1].scatter(z_valid[contam_valid], sev_valid[contam_valid],
                              marker=markers[catalog_slug], color=colors[catalog_slug],
                              label=labels[catalog_slug], alpha=0.6, s=50)
        
        # Plot 3: M200 vs redshift (highlight contaminated)
        if m200_valid is not None:
            # Non-contaminated
            axes[1, 0].scatter(z_valid[~contam_valid], m200_valid[~contam_valid],
                              marker=markers[catalog_slug], color=colors[catalog_slug],
                              alpha=0.3, s=30, label=f"{labels[catalog_slug]} (clean)")
            # Contaminated
            if np.any(contam_valid):
                axes[1, 0].scatter(z_valid[contam_valid], m200_valid[contam_valid],
                                  marker='X', color='red', alpha=0.7, s=80,
                                  label=f"{labels[catalog_slug]} (contaminated)", zorder=10)
        
        # Plot 4: Contamination severity histogram
        if np.any(contam_valid):
            axes[1, 1].hist(sev_valid[contam_valid], bins=20, alpha=0.6,
                           color=colors[catalog_slug], label=labels[catalog_slug], density=True)
    
    # Formatting
    axes[0, 0].set_xlabel('Redshift', fontsize=12)
    axes[0, 0].set_ylabel('Contamination Fraction', fontsize=12)
    axes[0, 0].set_title('Contamination Fraction vs Redshift', fontsize=14)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    axes[0, 0].set_ylim(0, 1)
    
    axes[0, 1].set_xlabel('Redshift', fontsize=12)
    axes[0, 1].set_ylabel('Contamination Severity', fontsize=12)
    axes[0, 1].set_title('Contamination Severity vs Redshift', fontsize=14)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    axes[0, 1].set_ylim(0, 1.1)
    
    axes[1, 0].set_xlabel('Redshift', fontsize=12)
    axes[1, 0].set_ylabel('M$_{200}$ (M$_{\\odot}$)', fontsize=12)
    axes[1, 0].set_title('M$_{200}$ vs Redshift (Contaminated Highlighted)', fontsize=14)
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3, which='both')
    axes[1, 0].legend(fontsize=10)
    
    axes[1, 1].set_xlabel('Contamination Severity', fontsize=12)
    axes[1, 1].set_ylabel('Normalized Density', fontsize=12)
    axes[1, 1].set_title('Contamination Severity Distribution', fontsize=14)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    output_path = output_dir / 'contamination_analysis.png'
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    logger.info(f"Saved contamination analysis to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    # Print statistics
    if all_redshift:
        all_redshift = np.array(all_redshift)
        all_contaminated = np.array(all_contaminated)
        all_severity = np.array(all_severity)
        
        high_z_mask = all_redshift >= 1.5
        n_high_z = np.sum(high_z_mask)
        n_contaminated = np.sum(all_contaminated[high_z_mask])
        
        print("\n" + "="*60)
        print("CONTAMINATION STATISTICS")
        print("="*60)
        print(f"High-z groups (z ≥ 1.5): {n_high_z}")
        print(f"Potentially contaminated: {n_contaminated} ({100*n_contaminated/n_high_z:.1f}%)")
        if n_contaminated > 0:
            print(f"Median severity: {np.median(all_severity[all_contaminated]):.3f}")
            print(f"Severity range: {np.min(all_severity[all_contaminated]):.3f} - {np.max(all_severity[all_contaminated]):.3f}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize projected contamination"
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='outputs/results',
        help='Results directory (default: outputs/results)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/figures',
        help='Output directory for plots (default: outputs/figures)'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display the plots'
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    
    # Load catalogs
    logger.info("Loading catalog data...")
    cw_all_table = load_catalog('cw_all', results_dir)
    cw_hcg_table = load_catalog('cw_hcg', results_dir)
    
    # Create plots
    plot_contamination_analysis(
        cw_all_table=cw_all_table,
        cw_hcg_table=cw_hcg_table,
        output_dir=output_dir,
        show=args.show
    )


if __name__ == '__main__':
    main()
