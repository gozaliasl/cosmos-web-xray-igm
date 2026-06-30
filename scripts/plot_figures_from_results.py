#!/usr/bin/env python3
"""
Regenerate analysis figures from saved results (no photometry or stacking).

Loads xray_catalog.csv (or .fits), optional stacking_results.fits, and X-ray maps,
then calls the same visualization functions used by main_analysis.py. Use this
to update figure styling or re-export after changing plot code without re-running
the full pipeline.

Usage:
    python plot_figures_from_results.py [--config config_refined_z.yaml] [--catalog CW-All] [--figures all]
    python plot_figures_from_results.py --config config.yaml --catalog CW-HCG --figures luminosity_redshift,diagnostics

Figure names: luminosity_redshift, detection_map, diagnostics, upper_limit_diagnostics,
              stacking_results, xray_map
Use --figures all to generate all that have required data.
"""

import sys
import re
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import yaml
import numpy as np
from astropy.table import Table

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from main_analysis import load_xray_maps_from_config
from xray_analysis.visualization import (
    plot_xray_map,
    plot_detection_map,
    plot_luminosity_redshift,
    plot_diagnostic_panel,
    plot_upper_limit_diagnostics,
    plot_stacking_results,
)
from xray_analysis.stacking import StackingResult


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def slugify(value: str) -> str:
    slug = re.sub(r'[^a-zA-Z0-9]+', '_', value.strip().lower()).strip('_')
    return slug or 'catalog'


def prepare_dirs(config: dict, slug: str) -> Dict[str, Path]:
    results_dir = Path(config['output']['results_dir']) / slug
    figures_dir = Path(config['output']['figures_dir']) / slug
    stacking_dir = Path(config['output']['stacking_dir']) / slug
    figures_dir.mkdir(parents=True, exist_ok=True)
    return {'results': results_dir, 'figures': figures_dir, 'stacking': stacking_dir}


def load_catalog(results_dir: Path) -> Optional[Table]:
    for name in ('xray_catalog.csv', 'xray_catalog.fits'):
        path = results_dir / name
        if path.exists():
            return Table.read(path)
    return None


def _column_to_bool(col) -> np.ndarray:
    """Convert a table column to boolean array. Handles CSV string 'True'/'False' (numpy bool would make both True)."""
    arr = np.asarray(col)
    if arr.dtype.kind == 'b':
        return arr
    if arr.dtype.kind == 'U' or arr.dtype.kind == 'S':
        return (arr == 'True') | (arr == 'true') | (arr == '1')
    return np.asarray(arr, dtype=bool)


def stacking_result_from_table(t: Table) -> StackingResult:
    """Reconstruct StackingResult from saved FITS table."""
    bin_edge_lower = np.asarray(t['bin_edge_lower'])
    bin_edge_upper = np.asarray(t['bin_edge_upper'])
    bin_edges = np.concatenate([[bin_edge_lower[0]], bin_edge_upper])
    median_properties = {}
    for col in t.colnames:
        if col.startswith('median_') and col not in ('median_redshift',):
            median_properties[col] = np.asarray(t[col])
    if 'median_redshift' in t.colnames:
        median_properties['median_redshift'] = np.asarray(t['median_redshift'])
    return StackingResult(
        bin_edges=bin_edges,
        bin_centers=np.asarray(t['bin_centers']),
        n_sources=np.asarray(t['n_sources'], dtype=int),
        stacked_signal=np.asarray(t['stacked_signal'], dtype=float),
        stacked_error=np.asarray(t['stacked_error'], dtype=float),
        snr=np.asarray(t['snr'], dtype=float),
        median_properties=median_properties,
        is_valid=np.asarray(t['is_valid'], dtype=bool),
        background_median=np.asarray(t['background_median'], dtype=float),
    )


def run_one_catalog(
    catalog_name: str,
    config: dict,
    dirs: Dict[str, Path],
    figures_wanted: List[str],
    xray_maps: Optional[dict],
    dpi: int,
    panel_label: Optional[str] = None,
) -> None:
    results_dir = dirs['results']
    figures_dir = dirs['figures']
    stacking_dir = dirs['stacking']

    table = load_catalog(results_dir)
    if table is None:
        print(f"  Skip {catalog_name}: no xray_catalog.csv/fits in {results_dir}")
        return

    n = len(table)
    redshift = np.asarray(table['Redshift'], dtype=float)
    luminosity = np.asarray(table['Luminosity_erg_s'], dtype=float)
    is_detected = _column_to_bool(table['Is_Detected'])
    upper_limits = np.asarray(table['Upper_Limit_Luminosity'], dtype=float)
    flagged_upper_limits = _column_to_bool(table['Is_Low_Upper_Limit'])
    suspected_fp = _column_to_bool(table['Is_Suspected_False_Positive']) if 'Is_Suspected_False_Positive' in table.colnames else np.zeros(n, dtype=bool)
    flux = np.asarray(table['Flux_erg_cm2_s'], dtype=float)
    upper_limit_flux = np.asarray(table['Upper_Limit_Flux_erg_cm2_s'], dtype=float) if 'Upper_Limit_Flux_erg_cm2_s' in table.colnames else np.full(n, np.nan)
    net_counts = np.asarray(table['Net_Counts'], dtype=float)
    snr = np.asarray(table['SNR'], dtype=float)
    ra = np.asarray(table['RA'], dtype=float)
    dec = np.asarray(table['DEC'], dtype=float)
    aperture_arcsec = np.asarray(table['Aperture_Arcsec'], dtype=float) if 'Aperture_Arcsec' in table.colnames else np.full(n, np.nan)

    stacking_redshift = None
    stacking_luminosity = None
    stacking_path = stacking_dir / 'stacking_results.fits'
    if stacking_path.exists():
        st = Table.read(stacking_path)
        if 'median_luminosity' in st.colnames:
            stacking_redshift = np.asarray(st['bin_centers'])
            stacking_luminosity = np.asarray(st['median_luminosity'])

    if 'luminosity_redshift' in figures_wanted:
        plot_luminosity_redshift(
            luminosity=luminosity,
            redshift=redshift,
            is_detected=is_detected,
            upper_limits=upper_limits,
            flagged_upper_limits=flagged_upper_limits,
            suspected_false_positives=suspected_fp,
            stacking_redshift=stacking_redshift,
            stacking_luminosity=stacking_luminosity,
            panel_label=panel_label,
            sample_label=catalog_name,
            output_path=figures_dir / 'luminosity_redshift.png',
            dpi=dpi,
        )
        print(f"  Wrote {figures_dir / 'luminosity_redshift.png'}")

    if 'diagnostics' in figures_wanted:
        plot_diagnostic_panel(
            net_counts=net_counts,
            snr=snr,
            luminosity=luminosity,
            redshift=redshift,
            is_detected=is_detected,
            output_path=figures_dir / 'diagnostics.png',
            dpi=dpi,
        )
        print(f"  Wrote {figures_dir / 'diagnostics.png'}")

    if 'upper_limit_diagnostics' in figures_wanted:
        plot_upper_limit_diagnostics(
            flux=flux,
            redshift=redshift,
            is_detected=is_detected,
            upper_limit_flux=upper_limit_flux,
            low_upper_limit_mask=flagged_upper_limits,
            output_path=figures_dir / 'upper_limit_diagnostics.png',
            dpi=dpi,
            sample_label=catalog_name,
        )
        print(f"  Wrote {figures_dir / 'upper_limit_diagnostics.png'}")

    if 'stacking_results' in figures_wanted and stacking_path.exists():
        stacking_result = stacking_result_from_table(Table.read(stacking_path))
        plot_stacking_results(
            stacking_result=stacking_result,
            output_path=figures_dir / 'stacking_results.png',
            dpi=dpi,
        )
        print(f"  Wrote {figures_dir / 'stacking_results.png'}")

    if xray_maps is not None:
        xray_map_viz = xray_maps.get('full') or xray_maps.get('single') or xray_maps.get('masked')
        median_ap = float(np.nanmedian(aperture_arcsec)) if np.any(np.isfinite(aperture_arcsec)) else 20.0

        if 'detection_map' in figures_wanted:
            plot_detection_map(
                xray_map=xray_map_viz,
                ra=ra,
                dec=dec,
                is_detected=is_detected,
                snr=snr,
                aperture_radius=median_ap,
                aperture_radii=aperture_arcsec if np.all(np.isfinite(aperture_arcsec)) else np.full(n, median_ap),
                output_path=figures_dir / 'detection_map.png',
                dpi=dpi,
            )
            print(f"  Wrote {figures_dir / 'detection_map.png'}")

        if 'xray_map' in figures_wanted:
            from xray_analysis.data_loader import load_group_catalog
            catalog_path = None
            for entry in config['data']['catalogs']:
                if entry['name'] == catalog_name:
                    catalog_path = Path(entry['group_catalog'])
                    break
            catalog = load_group_catalog(catalog_path) if catalog_path and catalog_path.exists() else None
            plot_xray_map(
                xray_map=xray_map_viz,
                catalog=catalog,
                output_path=figures_dir / 'xray_map.png',
                title=f"{catalog_name} - COSMOS X-ray Map",
                show_sources=True,
                dpi=dpi,
            )
            print(f"  Wrote {figures_dir / 'xray_map.png'}")


def main():
    parser = argparse.ArgumentParser(
        description='Regenerate figures from saved X-ray analysis results.',
        epilog='Figure names: luminosity_redshift, diagnostics, upper_limit_diagnostics, stacking_results, detection_map, xray_map. Use "all" for all available.',
    )
    parser.add_argument('--config', type=str, default='config_refined_z.yaml', help='Config file')
    parser.add_argument('--catalog', type=str, default=None, help='Catalog name (e.g. CW-All). If not set, process all catalogs in config.')
    parser.add_argument('--figures', type=str, default='all',
                        help='Comma-separated figure names or "all" (default: all)')
    parser.add_argument('--dpi', type=int, default=None, help='DPI (default: from config visualization.dpi)')
    parser.add_argument('--panel-labels', type=str, default=None,
                        help='Comma-separated panel labels for luminosity_redshift, e.g. "(a),(b)" when processing two catalogs in order')
    args = parser.parse_args()

    config = load_config(args.config)
    dpi = args.dpi or config.get('visualization', {}).get('dpi', 150)
    figures_wanted = [s.strip() for s in args.figures.split(',')]
    if figures_wanted == ['all']:
        figures_wanted = [
            'luminosity_redshift', 'diagnostics', 'upper_limit_diagnostics',
            'stacking_results', 'detection_map', 'xray_map',
        ]

    catalogs = [c for c in config['data']['catalogs'] if args.catalog is None or c['name'] == args.catalog]
    if not catalogs:
        print(f"No catalog matching --catalog {args.catalog}")
        return 1

    panel_labels = None
    if args.panel_labels:
        panel_labels = [s.strip() for s in args.panel_labels.split(',')]

    xray_maps = None
    if 'detection_map' in figures_wanted or 'xray_map' in figures_wanted:
        try:
            xray_maps = load_xray_maps_from_config(config)
        except Exception as e:
            print(f"Warning: could not load X-ray maps ({e}). Skipping detection_map and xray_map.")
            figures_wanted = [f for f in figures_wanted if f not in ('detection_map', 'xray_map')]

    for i, entry in enumerate(catalogs):
        name = entry['name']
        slug = slugify(name)
        dirs = prepare_dirs(config, slug)
        label = panel_labels[i] if panel_labels and i < len(panel_labels) else None
        print(f"Catalog: {name} ({slug})")
        run_one_catalog(
            catalog_name=name,
            config=config,
            dirs=dirs,
            figures_wanted=figures_wanted,
            xray_maps=xray_maps,
            dpi=dpi,
            panel_label=label,
        )
    print("Done.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
