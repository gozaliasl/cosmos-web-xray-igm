#!/usr/bin/env python3
"""
Compare X-ray properties from two background methods:
  - Annulus (local per-source background) vs redshift-binned median.

Use this to decide which technique is more reliable for:
  1. Individual group photometry (Lx, Flux, Net_Counts)
  2. Stacking analysis (stacked signal, SNR per redshift bin)

Photometry: one pipeline run already produces both (Luminosity_erg_s vs
Luminosity_Binned_erg_s) when background_use_redshift_binned_median is true.

Stacking: with stacking.background_method "bin_median" and stacking.save_local_comparison true,
a single pipeline run produces stacking_results.fits (primary = bin_median) and
stacking_results_local.fits (comparison). The comparison script uses those when present;
otherwise it looks for stacking_results_bin_median.fits + stacking_results_local.fits
(from --run-pipeline or manual two runs).

Usage:
  # Compare using existing outputs (photometry comparison only if binned columns exist)
  python compare_background_methods.py [--config config.yaml]

  # Run pipeline twice for stacking, then compare everything
  python compare_background_methods.py --config config.yaml --run-pipeline

  # Compare stacking from custom paths
  python compare_background_methods.py --config config.yaml \\
    --stacking-local outputs/stacking/cw_all/stacking_results_local.fits \\
    --stacking-bin-median outputs/stacking/cw_all/stacking_results_bin_median.fits
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import yaml
from astropy.table import Table


logger = logging.getLogger("compare_background_methods")


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "catalog"


def load_config(path: Path) -> dict:
    with path.open("r") as fh:
        return yaml.safe_load(fh)


def get_catalog_entries(config: dict) -> List[Dict]:
    data_cfg = config.get("data", {})
    entries = []
    if data_cfg.get("catalogs"):
        for entry in data_cfg["catalogs"]:
            catalog_path = entry.get("group_catalog")
            if not catalog_path:
                continue
            name = entry.get("name") or Path(catalog_path).stem
            entries.append({"name": name, "slug": slugify(name), "path": catalog_path})
    elif data_cfg.get("group_catalog"):
        path = data_cfg["group_catalog"]
        name = data_cfg.get("catalog_name") or Path(path).stem
        entries.append({"name": name, "slug": slugify(name), "path": path})
    return entries


def load_xray_catalog(results_dir: Path, slug: str) -> Optional[Table]:
    path = results_dir / slug / "xray_catalog.fits"
    if not path.exists():
        return None
    return Table.read(path)


def compare_photometry(
    table: Table,
    catalog_name: str,
    output_path: Path,
    show: bool = False,
) -> Tuple[Optional[dict], Optional[Path]]:
    """Compare annulus vs binned-median photometry (Lx, Flux, Net_Counts)."""
    if "Luminosity_Binned_erg_s" not in table.colnames:
        logger.warning(
            "Catalog %s has no Luminosity_Binned_erg_s; enable background_use_redshift_binned_median and re-run.",
            catalog_name,
        )
        return None, None

    lx = np.asarray(table["Luminosity_erg_s"], dtype=float)
    lx_b = np.asarray(table["Luminosity_Binned_erg_s"], dtype=float)
    flux = np.asarray(table["Flux_erg_cm2_s"], dtype=float)
    flux_b = np.asarray(table["Flux_Binned_erg_cm2_s"], dtype=float)
    net = np.asarray(table["Net_Counts"], dtype=float)
    net_b = np.asarray(table["Net_Counts_Binned"], dtype=float)

    # Finite positive for ratio/correlation
    mask_lx = np.isfinite(lx) & np.isfinite(lx_b) & (lx > 0) & (lx_b > 0)
    mask_flux = np.isfinite(flux) & np.isfinite(flux_b) & (flux > 0) & (flux_b > 0)
    mask_net = np.isfinite(net) & np.isfinite(net_b)

    stats_dict = {}
    for label, a, b, mask in [
        ("Lx", lx, lx_b, mask_lx),
        ("Flux", flux, flux_b, mask_flux),
        ("Net_Counts", net, net_b, mask_net),
    ]:
        if not np.any(mask):
            stats_dict[label] = {"n": 0, "corr": np.nan, "median_ratio": np.nan, "scatter": np.nan}
            continue
        x, y = a[mask], b[mask]
        if label == "Net_Counts":
            # Linear scale; avoid division by zero
            x, y = a[mask], b[mask]
            ratio_mask = (x != 0) & np.isfinite(x) & np.isfinite(y)
            if np.any(ratio_mask):
                ratio = float(np.median(y[ratio_mask] / x[ratio_mask]))
            else:
                ratio = np.nan
            corr = np.corrcoef(x, y)[0, 1] if len(x) > 1 else np.nan
            scatter = np.nan  # not in log space
        else:
            if np.any(x <= 0) or np.any(y <= 0):
                stats_dict[label] = {"n": int(np.sum(mask)), "corr": np.nan, "median_ratio": np.nan, "scatter": np.nan}
                continue
            log_x, log_y = np.log10(x), np.log10(y)
            corr = np.corrcoef(log_x, log_y)[0, 1]
            ratio = float(np.median(y / x))
            scatter = float(np.std(log_y - log_x))
        stats_dict[label] = {
            "n": int(np.sum(mask)),
            "corr": float(corr) if np.isfinite(corr) else np.nan,
            "median_ratio": float(ratio) if np.isfinite(ratio) else np.nan,
            "scatter": float(scatter) if np.isfinite(scatter) else np.nan,
        }

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=300)
    all_min = []
    all_max = []

    for ax, label, a, b, mask, xlabel, ylabel in [
        (axes[0], "Lx", lx, lx_b, mask_lx, r"$L_{\rm X}$ (annulus) [erg s$^{-1}$]", r"$L_{\rm X}$ (binned median) [erg s$^{-1}$]"),
        (axes[1], "Flux", flux, flux_b, mask_flux, r"Flux (annulus) [erg cm$^{-2}$ s$^{-1}$]", r"Flux (binned median)"),
        (axes[2], "Net_Counts", net, net_b, mask_net, "Net counts (annulus)", "Net counts (binned median)"),
    ]:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if not np.any(mask):
            ax.text(0.5, 0.5, "No valid pairs", ha="center", va="center", transform=ax.transAxes)
            continue
        x, y = a[mask], b[mask]
        valid = np.isfinite(x) & np.isfinite(y)
        if np.any(valid):
            x, y = x[valid], y[valid]
            if np.any(x > 0) and np.any(y > 0) and label != "Net_Counts":
                ax.loglog(x, y, "o", alpha=0.6, markersize=4, color="C0")
                lims = [min(x.min(), y.min()), max(x.max(), y.max())]
            else:
                ax.scatter(x, y, alpha=0.6, s=16, color="C0")
                lims = [min(x.min(), y.min()), max(x.max(), y.max())]
            all_min.append(lims[0])
            all_max.append(lims[1])
            lo, hi = max(lims[0] * 0.9, 1e-50 if label != "Net_Counts" else lims[0] * 0.9), lims[1] * 1.1
            # 1:1 line
            ax.plot([lo, hi], [lo, hi], "k--", alpha=0.7)
            # ±1σ and ±2σ bands for Lx and Flux (log-space scatter)
            s = stats_dict[label]
            if label != "Net_Counts" and np.isfinite(s.get("scatter", np.nan)) and s["scatter"] > 0:
                factor1 = 10 ** s["scatter"]
                factor2 = 10 ** (2 * s["scatter"])
                # 1σ band (thin)
                ax.plot([lo, hi], [lo * factor1, hi * factor1],
                        color="C1", linestyle="--", alpha=0.4, linewidth=1.0)
                ax.plot([lo, hi], [lo / factor1, hi / factor1],
                        color="C1", linestyle="--", alpha=0.4, linewidth=1.0)
                # 2σ band (fainter)
                ax.plot([lo, hi], [lo * factor2, hi * factor2],
                        color="C1", linestyle=":", alpha=0.3, linewidth=0.8)
                ax.plot([lo, hi], [lo / factor2, hi / factor2],
                        color="C1", linestyle=":", alpha=0.3, linewidth=0.8)
            ax.set_title(f"{label}: n={s['n']}, r={s['corr']:.3f}, ratio={s['median_ratio']:.3f}")
        # Publication-style: no gridlines, clear comparison line only
        ax.grid(False)

    # Add a single legend explaining the reference lines
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color="k", linestyle="--", label="1:1"),
        Line2D([0], [0], color="C1", linestyle="--", alpha=0.6, label=r"$\pm 1\sigma$"),
        Line2D([0], [0], color="C1", linestyle=":", alpha=0.6, label=r"$\pm 2\sigma$"),
    ]
    axes[0].legend(handles=legend_handles, fontsize=9, loc="upper left", frameon=True, framealpha=0.95)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    logger.info("Saved photometry comparison: %s", output_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return stats_dict, output_path


def load_stacking_results(path: Path) -> Optional[Table]:
    if not path.exists():
        return None
    return Table.read(path)


def compare_stacking(
    table_local: Table,
    table_bin_median: Table,
    catalog_name: str,
    output_path: Path,
    show: bool = False,
) -> Optional[dict]:
    """Compare stacking results: local (annulus) vs bin_median background."""
    if "bin_centers" not in table_local.colnames or "bin_centers" not in table_bin_median.colnames:
        logger.warning("Stacking tables missing bin_centers.")
        return None
    z_local = np.asarray(table_local["bin_centers"], dtype=float)
    z_bm = np.asarray(table_bin_median["bin_centers"], dtype=float)
    # Match by redshift bin (assume same binning)
    if len(z_local) != len(z_bm) or not np.allclose(z_local, z_bm):
        logger.warning("Stacking bin centers differ between local and bin_median; comparing by index.")
    n_bins = min(len(z_local), len(z_bm))
    sig_local = np.asarray(table_local["stacked_signal"], dtype=float)[:n_bins]
    sig_bm = np.asarray(table_bin_median["stacked_signal"], dtype=float)[:n_bins]
    snr_local = np.asarray(table_local["snr"], dtype=float)[:n_bins]
    snr_bm = np.asarray(table_bin_median["snr"], dtype=float)[:n_bins]
    z_common = z_local[:n_bins]

    valid = np.isfinite(sig_local) & np.isfinite(sig_bm) & (sig_local > 0) & (sig_bm > 0)
    if np.any(valid):
        log_sig_l = np.log10(sig_local[valid])
        log_sig_b = np.log10(sig_bm[valid])
        corr = np.corrcoef(log_sig_l, log_sig_b)[0, 1]
        ratio = np.median(sig_bm[valid] / sig_local[valid])
        scatter = np.std(log_sig_b - log_sig_l)
    else:
        corr = ratio = scatter = np.nan
    stats = {"n_bins": n_bins, "corr": float(corr), "median_ratio": float(ratio), "scatter": float(scatter)}

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=150)
    axes[0].scatter(sig_local, sig_bm, c=z_common, cmap="viridis", alpha=0.8)
    axes[0].set_xlabel("Stacked signal (local annulus)")
    axes[0].set_ylabel("Stacked signal (bin median)")
    lims_s = [min(sig_local.min(), sig_bm.min()), max(sig_local.max(), sig_bm.max())]
    axes[0].plot(lims_s, lims_s, "k--", alpha=0.7, label="1:1")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_title(f"Stacking: signal (n_bins={n_bins}, ratio={ratio:.3f})")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(snr_local, snr_bm, c=z_common, cmap="viridis", alpha=0.8)
    axes[1].set_xlabel("SNR (local annulus)")
    axes[1].set_ylabel("SNR (bin median)")
    lims_n = [min(snr_local.min(), snr_bm.min()), max(snr_local.max(), snr_bm.max())]
    axes[1].plot(lims_n, lims_n, "k--", alpha=0.7, label="1:1")
    axes[1].set_title("Stacking: SNR")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    logger.info("Saved stacking comparison: %s", output_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return stats


def run_pipeline_with_stacking_method(
    config_path: Path,
    config: dict,
    stacking_background_method: str,
    temp_config_path: Path,
) -> None:
    """Write config with given stacking.background_method and run main_analysis."""
    config_mod = dict(config)
    if "stacking" not in config_mod:
        config_mod["stacking"] = {}
    config_mod["stacking"] = dict(config_mod["stacking"])
    config_mod["stacking"]["background_method"] = stacking_background_method
    with open(temp_config_path, "w") as f:
        yaml.dump(config_mod, f, default_flow_style=False, sort_keys=False)
    script_path = Path(__file__).resolve().parent / "main_analysis.py"
    cmd = [sys.executable, "-u", str(script_path), "--config", str(temp_config_path)]
    logger.info("Running: python main_analysis.py --config ... (stacking background_method=%s)", stacking_background_method)
    subprocess.run(cmd, check=True, cwd=str(config_path.parent))


def copy_stacking_results_to_named(
    stacking_base: Path,
    catalog_entries: List[Dict],
    suffix: str,
) -> None:
    """Copy stacking_results.fits to stacking_results_<suffix>.fits per catalog."""
    for entry in catalog_entries:
        slug = entry["slug"]
        src = stacking_base / slug / "stacking_results.fits"
        dst = stacking_base / slug / f"stacking_results_{suffix}.fits"
        if src.exists():
            shutil.copy2(src, dst)
            logger.info("Copied %s -> %s", src, dst)
        else:
            logger.warning("No %s for catalog %s", src, entry["name"])


def write_summary(
    photometry_stats: Dict[str, Dict],
    stacking_stats: Dict[str, Dict],
    output_path: Path,
) -> None:
    """Write a short text summary to help decide which method is more reliable."""
    lines = [
        "# Background method comparison summary",
        "# Annulus = per-source local annulus; Binned = redshift-binned median.",
        "",
    ]
    lines.append("## Photometry (individual groups)")
    for catalog, stats in photometry_stats.items():
        if not stats:
            continue
        lines.append(f"  {catalog}:")
        for prop, s in stats.items():
            if isinstance(s, dict) and "n" in s and s["n"] > 0:
                lines.append(f"    {prop}: n={s['n']}, correlation={s.get('corr', np.nan):.3f}, "
                             f"median_ratio(binned/annulus)={s.get('median_ratio', np.nan):.3f}, scatter={s.get('scatter', np.nan):.3f}")
    lines.append("")
    lines.append("## Stacking")
    for catalog, s in stacking_stats.items():
        if s:
            lines.append(f"  {catalog}: n_bins={s.get('n_bins', 0)}, correlation={s.get('corr', np.nan):.3f}, "
                         f"median_ratio(bin_median/local)={s.get('median_ratio', np.nan):.3f}, scatter={s.get('scatter', np.nan):.3f}")
    lines.append("")
    lines.append("# Interpretation: median_ratio near 1 and high correlation suggest good agreement; "
                 "large scatter or ratio far from 1 may indicate one method is biased or noisier.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote summary: %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare X-ray properties from annulus vs redshift-binned median background methods.",
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument(
        "--run-pipeline",
        action="store_true",
        help="Run main_analysis twice (stacking local then bin_median) and save named stacking results before comparing.",
    )
    parser.add_argument(
        "--stacking-local",
        type=str,
        default=None,
        help="Path to stacking_results.fits (local/annulus). If set, --stacking-bin-median required for stacking comparison.",
    )
    parser.add_argument(
        "--stacking-bin-median",
        type=str,
        default=None,
        help="Path to stacking_results.fits (bin_median). Used with --stacking-local for single-catalog stacking comparison.",
    )
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    config = load_config(config_path)
    results_dir = Path(config["output"]["results_dir"]).resolve()
    stacking_dir = Path(config["output"]["stacking_dir"]).resolve()
    figures_dir = stacking_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    catalog_entries = get_catalog_entries(config)
    if not catalog_entries:
        raise ValueError("No catalogs in config.")

    if args.run_pipeline:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_local = tmp_path / "config_stacking_local.yaml"
            config_bm = tmp_path / "config_stacking_bin_median.yaml"
            run_pipeline_with_stacking_method(config_path, config, "local", config_local)
            copy_stacking_results_to_named(stacking_dir, catalog_entries, "local")
            run_pipeline_with_stacking_method(config_path, config, "bin_median", config_bm)
            copy_stacking_results_to_named(stacking_dir, catalog_entries, "bin_median")

    photometry_stats = {}
    stacking_stats = {}

    # Photometry comparison (annulus vs binned) per catalog
    for entry in catalog_entries:
        slug = entry["slug"]
        name = entry["name"]
        table = load_xray_catalog(results_dir, slug)
        if table is None:
            logger.warning("No xray_catalog for %s", name)
            continue
        out_photo = figures_dir / f"background_comparison_photometry_{slug}.png"
        stats, _ = compare_photometry(table, name, out_photo, show=args.show)
        if stats:
            photometry_stats[name] = stats

    # Stacking comparison
    if args.stacking_local and args.stacking_bin_median:
        t_local = load_stacking_results(Path(args.stacking_local).resolve())
        t_bm = load_stacking_results(Path(args.stacking_bin_median).resolve())
        if t_local is not None and t_bm is not None:
            cname = catalog_entries[0]["name"] if len(catalog_entries) == 1 else "stacking"
            out_stack = figures_dir / "background_comparison_stacking.png"
            stacking_stats[cname] = compare_stacking(t_local, t_bm, cname, out_stack, show=args.show)
    else:
        for entry in catalog_entries:
            slug = entry["slug"]
            name = entry["name"]
            path_local = stacking_dir / slug / "stacking_results_local.fits"
            # Primary is bin_median: stacking_results.fits; or from --run-pipeline: stacking_results_bin_median.fits
            path_bm = stacking_dir / slug / "stacking_results_bin_median.fits"
            if not path_bm.exists():
                path_bm = stacking_dir / slug / "stacking_results.fits"
            t_local = load_stacking_results(path_local)
            t_bm = load_stacking_results(path_bm)
            if t_local is None or t_bm is None:
                if t_local is None and t_bm is None:
                    logger.info(
                        "Stacking comparison skipped for %s: need stacking_results_local.fits and "
                        "stacking_results.fits (or stacking_results_bin_median.fits). "
                        "Run main_analysis with stacking.save_local_comparison: true.",
                        name,
                    )
                continue
            out_stack = figures_dir / f"background_comparison_stacking_{slug}.png"
            stacking_stats[name] = compare_stacking(t_local, t_bm, name, out_stack, show=args.show)

    summary_path = stacking_dir / "background_comparison_summary.txt"
    write_summary(photometry_stats, stacking_stats, summary_path)
    logger.info("Done. Summary: %s", summary_path)


if __name__ == "__main__":
    main()
