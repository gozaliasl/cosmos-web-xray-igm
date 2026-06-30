#!/usr/bin/env python3
"""
Plot M200 vs redshift for detected X-ray groups using M-Lx scaling relation.

Creates a plot showing M200(Lx) vs redshift for detected groups in both
cw_all and cw_hcg catalogs.
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


def load_detections(catalog_slug: str, results_dir: Path) -> Optional[Table]:
    """Load detection data for a catalog."""
    # Try detections.csv first (detected only), then xray_catalog.csv
    detections_path = results_dir / catalog_slug / "detections.csv"
    if not detections_path.exists():
        catalog_path = results_dir / catalog_slug / "xray_catalog.csv"
        if not catalog_path.exists():
            logger.warning(f"No detection data found for {catalog_slug}")
            return None
        logger.info(f"Loading from {catalog_path.name}")
        table = Table.read(catalog_path)
        # Filter for detected groups
        if 'Is_Detected' in table.colnames:
            table = table[table['Is_Detected'] == True]
        else:
            logger.warning(f"No Is_Detected column in {catalog_path.name}")
            return None
    else:
        logger.info(f"Loading from {detections_path.name}")
        table = Table.read(detections_path)
    
    return table


def plot_m200_redshift(
    cw_all_table: Optional[Table],
    cw_hcg_table: Optional[Table],
    output_path: Path,
    figsize: tuple = (6, 5),
    dpi: int = 300,
    show: bool = False
):
    """
    Plot M200 vs redshift for detected groups.
    
    Parameters
    ----------
    cw_all_table : Table or None
        Detection table for CW-All catalog
    cw_hcg_table : Table or None
        Detection table for CW-HCG catalog
    output_path : Path
        Output file path for the plot
    figsize : tuple
        Figure size (width, height)
    dpi : int
        Figure resolution
    show : bool
        Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Use high-level matplotlib tab colors (Tableau palette)
    colors = {'cw_all': 'tab:blue', 'cw_hcg': 'tab:orange'}
    markers = {'cw_all': 'o', 'cw_hcg': 's'}
    labels = {'cw_all': 'CW-All', 'cw_hcg': 'CW-HCG'}
    
    plotted_any = False
    
    for catalog_slug, table in [('cw_all', cw_all_table), ('cw_hcg', cw_hcg_table)]:
        if table is None:
            continue
        
        # Extract data
        if 'M200_Luminosity_Msun' not in table.colnames:
            logger.warning(f"No M200_Luminosity_Msun column in {catalog_slug}")
            continue
        
        if 'Redshift' not in table.colnames:
            logger.warning(f"No Redshift column in {catalog_slug}")
            continue
        
        redshift = np.array(table['Redshift'], dtype=float)
        m200 = np.array(table['M200_Luminosity_Msun'], dtype=float)
        
        # Get errors if available
        if 'M200_Luminosity_Error' in table.colnames:
            m200_err = np.array(table['M200_Luminosity_Error'], dtype=float)
        else:
            m200_err = None
        
        # Filter valid data
        valid = (
            np.isfinite(redshift) & 
            np.isfinite(m200) & 
            (m200 > 0) & 
            (redshift > 0)
        )
        
        if m200_err is not None:
            valid &= np.isfinite(m200_err) & (m200_err > 0)
        
        if not np.any(valid):
            logger.warning(f"No valid data for {catalog_slug}")
            continue
        
        z_valid = redshift[valid]
        m200_valid = m200[valid]
        m200_err_valid = m200_err[valid] if m200_err is not None else None
        
        # Plot with error bars if available
        if m200_err_valid is not None:
            ax.errorbar(
                z_valid,
                m200_valid,
                yerr=m200_err_valid,
                fmt=markers[catalog_slug],
                color=colors[catalog_slug],
                label=labels[catalog_slug],
                alpha=0.8,
                capsize=2,
                capthick=1.0,
                markersize=4,
                linewidth=1.2,
                elinewidth=0.9,
            )
        else:
            ax.scatter(
                z_valid,
                m200_valid,
                marker=markers[catalog_slug],
                color=colors[catalog_slug],
                label=labels[catalog_slug],
                alpha=0.8,
                s=24
            )
        
        plotted_any = True
        logger.info(f"Plotted {np.sum(valid)} detections for {catalog_slug}")
    
    if not plotted_any:
        logger.error("No data to plot!")
        plt.close(fig)
        return
    
    # Formatting: publication-style
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
    ax.tick_params(axis='both', which='major', labelsize=11, width=1.0, length=4)

    ax.set_xlabel('Redshift', fontsize=12)
    ax.set_ylabel(r'$M_{200}$ (M$_\odot$)', fontsize=12)
    ax.set_yscale('log')
    ax.grid(False)
    leg = ax.legend(loc='best', fontsize=10, frameon=True, framealpha=0.95, edgecolor='#cccccc')
    
    # Set reasonable axis limits
    if plotted_any:
        ax.set_xlim(left=0)
        # Y-axis will auto-scale with log scale
    
    plt.tight_layout()
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    logger.info(f"Saved plot to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot M200 vs redshift for detected X-ray groups"
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='outputs/results',
        help='Results directory (default: outputs/results)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/figures/m200_redshift_detections.png',
        help='Output file path (default: outputs/figures/m200_redshift_detections.png)'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display the plot'
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_path = Path(args.output)
    
    # Load detection data
    logger.info("Loading detection data...")
    cw_all_table = load_detections('cw_all', results_dir)
    cw_hcg_table = load_detections('cw_hcg', results_dir)
    
    # Create plot
    plot_m200_redshift(
        cw_all_table=cw_all_table,
        cw_hcg_table=cw_hcg_table,
        output_path=output_path,
        show=args.show
    )


if __name__ == '__main__':
    main()
