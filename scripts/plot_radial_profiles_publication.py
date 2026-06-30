"""
Publication-quality 4-panel stacked X-ray radial surface-brightness figure.

Shows non-detected CW-All groups stacked in 4 redshift bins. Each panel has:
  - Shaded 1σ confidence band (fill_between) + solid profile line
  - XMM-Newton PSF model (orange dashed)
  - Beta-model fit (green dash-dot)
  - Random-position null stack (grey dotted)
  - Background level (horizontal red dashed)
  - R500 vertical marker (purple)
  - Primary x-axis: r / R500; secondary top x-axis: physical kpc

Usage
-----
  # First run: compute profiles and cache to NPZ (~5–10 min)
  python plot_radial_profiles_publication.py --config config.yaml --catalog CW-All

  # Subsequent runs: read cached NPZ, re-plot only (< 5 s)
  python plot_radial_profiles_publication.py --config config.yaml --catalog CW-All --plot-only

  # Override output file
  python plot_radial_profiles_publication.py --config config.yaml --out figures/my_fig.pdf
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from astropy.io import fits
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM

# ── project imports ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
if (ROOT / "src").exists():
    sys.path.insert(0, str(ROOT / "src"))

from stack_radial_profiles import (
    load_config,
    build_cosmology,
    _stack_one_redshift_bin,
    _fit_and_plot_beta_model,
    _beta_model_profile,
    _stacked_annulus_background,
    generate_xmm_psf_profile,
    load_stacking_weights,
)
from main_analysis import load_xray_maps_from_config
from xray_analysis.data_loader import load_group_catalog

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("pub_profiles")

# ── aesthetics ─────────────────────────────────────────────────────────────────
PROFILE_COLOR  = "#1f77b4"   # blue
PSF_COLOR      = "#d62728"   # red (more visible than orange)
BETA_COLOR     = "#2ca02c"   # green
RANDOM_COLOR   = "#7f7f7f"   # grey
BG_COLOR       = "#9467bd"   # purple
R500_COLOR     = "#ff7f0e"   # orange

PROFILE_ALPHA  = 0.25        # fill_between transparency
LINEWIDTH      = 1.8

# ── non-detection z bins (4 bins matching the existing pipeline output) ────────
NONDET_ZBINS = [
    (0.05, 0.85),
    (0.85, 1.30),
    (1.30, 2.20),
    (2.20, 3.80),
]

# ── helpers ────────────────────────────────────────────────────────────────────

def _kpc_per_arcsec(cosmo: FlatLambdaCDM, z: float) -> float:
    return cosmo.kpc_proper_per_arcmin(z).value / 60.0


def _load_detection_flag(results_dir: Path, slug: str) -> Optional[np.ndarray]:
    """Return boolean array marking detected groups (True = detected) or None."""
    for fname in (f"stacking_results_{slug}.fits", "stacking_results.fits",
                  f"xray_catalog_{slug}.fits", "xray_catalog.fits"):
        p = results_dir / fname
        if p.exists():
            t = Table.read(p)
            for col in ("is_detected", "IS_DETECTED", "detected", "DETECTED"):
                if col in t.colnames:
                    return np.asarray(t[col], dtype=bool)
    return None


def compute_and_cache(
    config: dict,
    catalog_name: str,
    cache_dir: Path,
    max_radius_r500: float = 2.5,
    bin_width_arcsec: float = 2.0,
) -> List[dict]:
    """
    Run stacking for each NONDET_ZBINS bin on non-detected groups, cache to NPZ.
    Returns list of dicts with keys:
        z_min, z_max, z_med, n_groups, r_kpc, sb, sb_err, sb_random,
        r500_med_kpc, r200_med_kpc, bg_sb
    """
    cosmo = build_cosmology(config)
    xray_maps = load_xray_maps_from_config(config)
    use_masked = config.get("stacking", {}).get("use_masked_map", False)
    xray_map = (xray_maps.get("masked") if use_masked else None) or \
               xray_maps.get("full") or xray_maps.get("single")
    image = xray_map.data
    wcs   = xray_map.wcs

    # Load group catalog
    slug = catalog_name.lower().replace(" ", "_").replace("-", "_")
    cat_entry = next(
        c for c in config["analysis"]["catalogs"] if c["name"] == catalog_name
    )
    catalog = load_group_catalog(str(Path(cat_entry["group_catalog"]).resolve()))
    ra  = np.asarray(catalog.get_coordinates()[0], dtype=float)
    dec = np.asarray(catalog.get_coordinates()[1], dtype=float)
    z   = np.asarray(catalog.get_redshifts(),      dtype=float)

    # Filter to non-detections
    results_dir = Path(config["output"]["results_dir"]).resolve()
    det_flag = _load_detection_flag(results_dir, slug)
    if det_flag is not None and len(det_flag) == len(z):
        is_nondet = ~det_flag
        log.info("Non-detections: %d / %d", is_nondet.sum(), len(z))
    else:
        log.warning("Could not load detection flags; using all groups")
        is_nondet = np.ones(len(z), dtype=bool)

    ra_nd  = ra[is_nondet]
    dec_nd = dec[is_nondet]
    z_nd   = z[is_nondet]

    weights = load_stacking_weights(results_dir, slug, ra_nd, dec_nd, config)

    cache_dir.mkdir(parents=True, exist_ok=True)
    bins_data = []

    for z_min, z_max in NONDET_ZBINS:
        cache_file = cache_dir / f"profile_{slug}_nondet_z{z_min:.2f}_{z_max:.2f}.npz"

        if cache_file.exists():
            log.info("Loading cached profile: %s", cache_file.name)
            d = np.load(cache_file, allow_pickle=True)
            bins_data.append({k: d[k] for k in d.files})
            continue

        mask_bin = (z_nd >= z_min) & (z_nd < z_max)
        n = int(mask_bin.sum())
        if n < 3:
            log.warning("Bin z=%.2f-%.2f: only %d groups, skipping.", z_min, z_max, n)
            continue

        z_med   = float(np.median(z_nd[mask_bin]))
        kpc_as  = _kpc_per_arcsec(cosmo, z_med)

        # Estimate rough R500 for this bin from the FITS summary if available
        r500_med_kpc = 250.0  # conservative default
        fits_sum = results_dir / slug / "stacking_radial_results_nondet.fits"
        if fits_sum.exists():
            t = Table.read(fits_sum)
            m = (np.abs(t["z_min"] - z_min) < 0.01) & (np.abs(t["z_max"] - z_max) < 0.01)
            if np.any(m):
                r500_med_kpc = float(t["r500_median_kpc"][m][0])
                r200_med_kpc = float(t["r200_median_kpc"][m][0])
        else:
            r200_med_kpc = r500_med_kpc * 1.4

        max_radius_arcsec = (r500_med_kpc * max_radius_r500) / kpc_as
        log.info(
            "Bin z=%.2f-%.2f  N=%d  z_med=%.3f  R500=%.0f kpc  max_r=%.1f\"",
            z_min, z_max, n, z_med, r500_med_kpc, max_radius_arcsec,
        )

        w_bin = weights[mask_bin] if weights is not None else None
        r_kpc, sb, sb_err, sb_random, sb_psf, area_accum = _stack_one_redshift_bin(
            image=image,
            wcs=wcs,
            ra_bin=ra_nd[mask_bin],
            dec_bin=dec_nd[mask_bin],
            z_bin=z_nd[mask_bin],
            weights_bin=w_bin,
            max_radius_arcsec=max_radius_arcsec,
            bin_width_arcsec=bin_width_arcsec,
            cosmo=cosmo,
            exposure_map=None,
        )

        # Background from outer annulus (1.5–2.5 R500 in kpc)
        bg_sb = _stacked_annulus_background(
            r_kpc, sb,
            background_inner_kpc=r500_med_kpc * 1.5,
            background_outer_kpc=r500_med_kpc * 2.5,
        )

        payload = dict(
            z_min=np.float64(z_min), z_max=np.float64(z_max),
            z_med=np.float64(z_med), n_groups=np.int64(n),
            r_kpc=r_kpc, sb=sb, sb_err=sb_err, sb_random=sb_random,
            sb_psf=sb_psf, area_accum=area_accum,
            r500_med_kpc=np.float64(r500_med_kpc),
            r200_med_kpc=np.float64(r200_med_kpc),
            bg_sb=np.float64(bg_sb if (bg_sb is not None and np.isfinite(bg_sb)) else np.nan),
        )
        np.savez(cache_file, **payload)
        log.info("Cached → %s", cache_file)
        bins_data.append(payload)

    return bins_data


def _psf_model(r_kpc: np.ndarray, z_med: float, cosmo: FlatLambdaCDM,
               sb_peak: float) -> np.ndarray:
    """XMM PSF model normalised to sb_peak at r=0."""
    psf = generate_xmm_psf_profile(r_kpc, z_med, cosmo)
    if psf[0] > 0 and np.isfinite(psf[0]):
        psf = psf * (sb_peak / psf[0])
    return psf


def _draw_panel(
    ax: plt.Axes,
    r_kpc: np.ndarray,
    sb: np.ndarray,
    sb_err: np.ndarray,
    sb_random: np.ndarray,
    r500_med_kpc: float,
    z_med: float,
    z_min: float,
    z_max: float,
    n_groups: int,
    bg_sb: float,
    cosmo: FlatLambdaCDM,
    is_bottom: bool,
    is_left: bool,
    show_legend: bool,
    add_kpc_axis: bool,
) -> None:
    """Draw one radial profile panel in r/R500 units."""

    r_norm = r_kpc / r500_med_kpc  # x axis: r/R500

    # ── valid range ───────────────────────────────────────────────────────────
    valid = np.isfinite(sb) & (sb > 0) & np.isfinite(sb_err) & (sb_err > 0)
    if not np.any(valid):
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        return

    idx = np.where(valid)[0]
    i0, i1 = int(idx[0]), int(idx[-1]) + 1
    xv   = r_norm[i0:i1]
    yv   = sb[i0:i1]
    yerr = sb_err[i0:i1]

    # ── background band ───────────────────────────────────────────────────────
    if np.isfinite(bg_sb) and bg_sb > 0:
        ax.axhline(bg_sb, color=BG_COLOR, lw=1.2, ls="--", alpha=0.7, zorder=1)
        ax.axhspan(bg_sb * 0.8, bg_sb * 1.2, color=BG_COLOR, alpha=0.07, zorder=0)

    # ── random null test ──────────────────────────────────────────────────────
    rand_valid = np.isfinite(sb_random) & (sb_random > 0)
    if np.any(rand_valid):
        xr = r_norm[rand_valid]
        yr = sb_random[rand_valid]
        ax.plot(xr, yr, color=RANDOM_COLOR, lw=1.0, ls=":", alpha=0.7,
                zorder=2, label="Random stack (null)")

    # ── stacked profile: shaded band + line ───────────────────────────────────
    ax.fill_between(xv, yv - yerr, yv + yerr,
                    color=PROFILE_COLOR, alpha=PROFILE_ALPHA, zorder=3)
    ax.plot(xv, yv, color=PROFILE_COLOR, lw=LINEWIDTH, zorder=4,
            label="Stacked SB")

    # ── XMM PSF ───────────────────────────────────────────────────────────────
    psf = _psf_model(r_kpc, z_med, cosmo, sb_peak=float(yv[0]))
    psf_valid = np.isfinite(psf) & (psf > 0)
    if np.any(psf_valid):
        ax.plot(r_norm[psf_valid], psf[psf_valid],
                color=PSF_COLOR, lw=LINEWIDTH, ls="--", zorder=5,
                label="XMM-Newton PSF")

    # ── beta-model fit ────────────────────────────────────────────────────────
    # Use a dummy invisible axes to absorb internal ax.plot calls; extract params only
    beta_fit = None
    if r500_med_kpc > 0:
        _fig_dummy, _ax_dummy = plt.subplots(1, 1)
        try:
            beta_fit = _fit_and_plot_beta_model(
                ax=_ax_dummy,
                r_kpc=r_kpc,
                sb_stack=sb,
                sb_stack_err=sb_err,
                r500_med_kpc=r500_med_kpc,
                x_plot=r_norm,
            )
        finally:
            plt.close(_fig_dummy)

    if beta_fit is not None:
        beta_med    = beta_fit.get("beta_med")
        s0_med      = beta_fit.get("s0_med")
        r_core_kpc  = beta_fit.get("r_core_kpc")
        if (beta_med is not None and s0_med is not None and r_core_kpc is not None
                and all(np.isfinite([beta_med, s0_med, r_core_kpc]))):
            sb_beta = _beta_model_profile(r_kpc, s0_med, r_core_kpc, beta_med)
            beta_valid = np.isfinite(sb_beta) & (sb_beta > 0)
            ax.plot(r_norm[beta_valid], sb_beta[beta_valid],
                    color=BETA_COLOR, lw=LINEWIDTH, ls="-.", zorder=6,
                    label=rf"$\beta$-model ($\beta$={beta_med:.2f})")

    # ── R500 vertical line ────────────────────────────────────────────────────
    ax.axvline(1.0, color=R500_COLOR, lw=1.2, ls=":", alpha=0.85, zorder=7,
               label=r"$R_{500}$")

    # ── log scale + limits ────────────────────────────────────────────────────
    yp = yv[yv > 0]
    if len(yp) > 0:
        ylo = np.nanmin(yp) / 3.0
        yhi = np.nanmax(yp) * 5.0
        ax.set_yscale("log")
        ax.set_ylim(max(ylo, 1e-12), yhi)
    ax.set_xlim(0.0, 2.5)

    # ── annotations ──────────────────────────────────────────────────────────
    label_txt = (
        f"$z = {z_min:.2f}$–${z_max:.2f}$\n"
        f"$\\langle z \\rangle = {z_med:.2f}$\n"
        f"$N = {n_groups}$"
    )
    ax.text(0.97, 0.97, label_txt, transform=ax.transAxes,
            ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.85))

    # ── secondary kpc axis on top ─────────────────────────────────────────────
    if add_kpc_axis:
        ax2 = ax.twiny()
        kpc_ticks = np.array([50, 100, 200, 300, 500, 750])
        kpc_in_r500 = kpc_ticks / r500_med_kpc
        inside = (kpc_in_r500 >= 0) & (kpc_in_r500 <= 2.5)
        ax2.set_xlim(0.0, 2.5)
        ax2.set_xticks(kpc_in_r500[inside])
        ax2.set_xticklabels([str(k) for k in kpc_ticks[inside]], fontsize=7)
        ax2.set_xlabel("Radius (kpc)", fontsize=8, labelpad=3)
        ax2.tick_params(axis="x", length=3, width=0.8)

    # ── axis labels ───────────────────────────────────────────────────────────
    if is_bottom:
        ax.set_xlabel(r"$r\ /\ R_{500}$", fontsize=11)
    if is_left:
        ax.set_ylabel(
            r"Surface brightness (counts s$^{-1}$ pixel$^{-1}$)", fontsize=9
        )

    # ── legend (only in first panel) ─────────────────────────────────────────
    if show_legend:
        handles = [
            Line2D([0], [0], color=PROFILE_COLOR, lw=LINEWIDTH, label="Stacked SB"),
            Patch(facecolor=PROFILE_COLOR, alpha=PROFILE_ALPHA, label=r"$1\sigma$ band"),
            Line2D([0], [0], color=PSF_COLOR,    lw=LINEWIDTH, ls="--", label="XMM PSF"),
            Line2D([0], [0], color=BETA_COLOR,   lw=LINEWIDTH, ls="-.", label=r"$\beta$-model"),
            Line2D([0], [0], color=RANDOM_COLOR, lw=1.0,       ls=":",  label="Random (null)"),
            Line2D([0], [0], color=BG_COLOR,     lw=1.2,       ls="--", label="Background"),
            Line2D([0], [0], color=R500_COLOR,   lw=1.2,       ls=":",  label=r"$R_{500}$"),
        ]
        ax.legend(handles=handles, fontsize=7.5, frameon=True,
                  loc="upper right", framealpha=0.9,
                  edgecolor="0.7", handlelength=2.0)

    ax.tick_params(axis="both", which="major", labelsize=9)
    ax.tick_params(axis="both", which="minor", labelsize=7)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def make_figure(
    bins_data: List[dict],
    cosmo: FlatLambdaCDM,
    catalog_name: str,
    out_path: Path,
    dpi: int = 300,
) -> None:
    """Render 2×2 publication figure."""
    n_bins = len(bins_data)
    ncols = 2
    nrows = (n_bins + 1) // 2

    fig = plt.figure(figsize=(10, nrows * 4.5), dpi=dpi)
    gs  = gridspec.GridSpec(nrows, ncols, figure=fig,
                            hspace=0.35, wspace=0.12)

    for idx, d in enumerate(bins_data):
        row, col = divmod(idx, ncols)
        ax = fig.add_subplot(gs[row, col])

        _draw_panel(
            ax=ax,
            r_kpc      = d["r_kpc"],
            sb         = d["sb"],
            sb_err     = d["sb_err"],
            sb_random  = d["sb_random"],
            r500_med_kpc = float(d["r500_med_kpc"]),
            z_med      = float(d["z_med"]),
            z_min      = float(d["z_min"]),
            z_max      = float(d["z_max"]),
            n_groups   = int(d["n_groups"]),
            bg_sb      = float(d["bg_sb"]),
            cosmo      = cosmo,
            is_bottom  = (row == nrows - 1),
            is_left    = (col == 0),
            show_legend= (idx == 0),
            add_kpc_axis = (row == 0),
        )

    fig.suptitle(
        f"Stacked X-ray Surface Brightness Profiles — {catalog_name} (non-detections)",
        fontsize=12, y=1.01,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    for ext in (out_path.suffix,):
        fig.savefig(out_path, bbox_inches="tight", dpi=dpi)
    # Also save PDF
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    log.info("Saved: %s  and  %s", out_path, pdf_path)
    plt.close(fig)


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",    default="config.yaml",  help="Pipeline YAML config")
    parser.add_argument("--catalog",   default="CW-All",       help="Catalog name in config")
    parser.add_argument("--out",       default=None,            help="Output figure path")
    parser.add_argument("--cache-dir", default=None,            help="NPZ cache directory")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip stacking; load from NPZ cache only")
    parser.add_argument("--max-r500",  type=float, default=2.5,
                        help="Maximum radius in units of R500 (default 2.5)")
    parser.add_argument("--bin-width", type=float, default=2.0,
                        help="Radial bin width in arcsec (default 2.0)")
    parser.add_argument("--dpi",       type=int,   default=300)
    args = parser.parse_args()

    config = load_config(Path(args.config))
    cosmo  = build_cosmology(config)

    slug      = args.catalog.lower().replace(" ", "_").replace("-", "_")
    cache_dir = Path(args.cache_dir) if args.cache_dir else \
                Path(config["output"]["results_dir"]) / slug / "profile_cache"
    out_path  = Path(args.out) if args.out else \
                Path(config["output"]["results_dir"]) / slug / \
                "figures" / "stacking_radial_profiles_publication.png"

    if args.plot_only:
        # Load from cache only
        bins_data = []
        for z_min, z_max in NONDET_ZBINS:
            p = cache_dir / f"profile_{slug}_nondet_z{z_min:.2f}_{z_max:.2f}.npz"
            if p.exists():
                d = np.load(p, allow_pickle=True)
                bins_data.append({k: d[k] for k in d.files})
            else:
                log.warning("Cache missing: %s — run without --plot-only first", p)
    else:
        bins_data = compute_and_cache(
            config       = config,
            catalog_name = args.catalog,
            cache_dir    = cache_dir,
            max_radius_r500 = args.max_r500,
            bin_width_arcsec = args.bin_width,
        )

    if not bins_data:
        log.error("No profile data available. Check cache or re-run stacking.")
        sys.exit(1)

    make_figure(bins_data, cosmo, args.catalog, out_path, dpi=args.dpi)


if __name__ == "__main__":
    main()
