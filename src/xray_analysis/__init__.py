"""
X-ray Analysis Package for Galaxy Groups

This package provides tools for analyzing X-ray emission from galaxy groups
using aperture photometry, stacking analysis, and spectral modeling.
"""

__version__ = "0.1.0"
__author__ = "Galaxy Groups X-ray Analysis"

from .data_loader import load_group_catalog, load_xray_maps
from .photometry import perform_aperture_photometry
from .detection import calculate_detection_significance
from .xray_properties import calculate_xray_luminosity, calculate_xray_flux
from .mass_estimation import (
    estimate_mass_from_temperature,
    estimate_mass_from_luminosity,
    estimate_mass_iterative
)
from .stacking import perform_stacking_analysis
from .visualization import plot_xray_map, plot_detection_map, plot_luminosity_redshift

__all__ = [
    'load_group_catalog',
    'load_xray_maps',
    'perform_aperture_photometry',
    'calculate_detection_significance',
    'calculate_xray_luminosity',
    'calculate_xray_flux',
    'estimate_mass_from_temperature',
    'estimate_mass_from_luminosity',
    'estimate_mass_iterative',
    'perform_stacking_analysis',
    'plot_xray_map',
    'plot_detection_map',
    'plot_luminosity_redshift'
]
