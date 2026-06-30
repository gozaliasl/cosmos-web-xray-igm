# COSMOS-Web X-ray IGM — Paper I

**A Deep X-ray Catalog of Galaxy Groups: Characterizing the Hot Intragroup Medium in COSMOS-Web to z=3.7**

Ghassem Gozaliasl (Aalto University / University of Helsinki)

## Overview

This project delivers:
- Uniform aperture-photometry X-ray pipeline for 1,678 COSMOS-Web galaxy groups (CW-All) and 912 Hickson compact groups (CW-HCG)
- Individual X-ray detections (SNR ≥ 2), upper limits, luminosities, temperatures, halo masses
- X-ray catalog `xray_catalog.fits` ready for further scientific exploitation
- Figures and LaTeX source for A&A Paper I submission

## Structure

```
cosmos-web-xray-igm/
├── src/xray_analysis/      # Core pipeline library
│   ├── photometry.py       # Aperture photometry & background subtraction
│   ├── detection.py        # SNR thresholding & upper limits
│   ├── stacking.py         # X-ray stacking
│   ├── spectral_model.py   # Flux conversion & K-correction
│   ├── mass_estimation.py  # Scaling relations (LX→T→M200)
│   ├── peak_finding.py     # X-ray centroid refinement
│   ├── contamination.py    # Point-source & neighbour masking
│   ├── data_loader.py      # FITS map I/O
│   ├── visualization.py    # Plotting utilities
│   └── xray_properties.py  # Main properties class
├── scripts/
│   ├── main_analysis.py    # Master pipeline entry point
│   ├── run_snr_sweep.py    # SNR threshold sensitivity sweep
│   ├── plot_figures_from_results.py
│   ├── create_xray_map_plot.py
│   └── ...
├── configs/
│   ├── config.yaml         # Default pipeline config
│   └── config_refined_z.yaml
├── paper/
│   ├── cosmos-web_xray_paper.tex
│   ├── cosmos-web_xray_paper_standalone.tex
│   ├── references.bib
│   ├── aa.cls / aa.bst
│   ├── compile_paper.sh
│   └── figures/
├── alexis_ghassem_measurments/   # Cross-validation with Finoguenov measurements
└── outputs/                # Pipeline outputs (not in git)
```

## Data (not in git — see ../data/)

- `../data/xray-map/cosmos_chaxmm14_520*.fits` — Chandra+XMM-Newton mosaics
- `../data/group-catalog/cosmos_web_groups_catalog_refined_z.fits` — COSMOS-Web group catalog
- `../data/specz/Webb_Specz_Feb2026.fits` — Spectroscopic redshifts

## Running the pipeline

```bash
python scripts/main_analysis.py --config configs/config.yaml
```

## Paper

Compile with:
```bash
cd paper && bash compile_paper.sh
```

## References

- Toni et al. 2025, A&A, 697, A197 — COSMOS-Web group catalog
- Gozaliasl et al. 2019, MNRAS, 483, 3545 — COSMOS X-ray groups
- Leauthaud et al. 2010, ApJ, 709, 97 — M–LX scaling relation
- Kettula et al. 2015, MNRAS, 451, 1460 — LX–T relation
