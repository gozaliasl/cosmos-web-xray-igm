"""
Data loading and validation module.

Handles loading of galaxy group catalogs and X-ray maps with proper
error handling and validation.
"""

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from pathlib import Path
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class GroupCatalog:
    """Container for galaxy group catalog data."""

    def __init__(self, data: Table, filepath: str):
        self.data = data
        self.filepath = filepath
        self.n_groups = len(data)

    def __repr__(self):
        return f"GroupCatalog(n_groups={self.n_groups}, file='{self.filepath}')"

    def get_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get RA and Dec coordinates."""
        # Try both possible column names
        ra_col = 'Ra' if 'Ra' in self.data.colnames else 'RA'
        dec_col = 'Dec' if 'Dec' in self.data.colnames else 'DEC'
        return self.data[ra_col], self.data[dec_col]

    def get_redshifts(self) -> np.ndarray:
        """Get redshifts."""
        # Try both possible column names
        z_col = 'z' if 'z' in self.data.colnames else 'Redshift'
        return self.data[z_col]

    def get_properties(self) -> Dict[str, np.ndarray]:
        """Get all group properties as dictionary."""
        return {col: self.data[col] for col in self.data.colnames}


class XrayMap:
    """Container for X-ray map data with WCS information."""

    def __init__(self, data: np.ndarray, error: np.ndarray,
                 wcs: WCS, header: fits.Header, filepath: str):
        self.data = data
        self.error = error
        self.wcs = wcs
        self.header = header
        self.filepath = filepath
        self.shape = data.shape

        # Get pixel scale in arcsec/pixel
        try:
            self.pixel_scale = abs(header['CDELT1']) * 3600  # degrees to arcsec
        except KeyError:
            # Try alternative keywords
            self.pixel_scale = abs(wcs.wcs.cdelt[0]) * 3600

    def __repr__(self):
        return (f"XrayMap(shape={self.shape}, "
                f"pixel_scale={self.pixel_scale:.2f} arcsec/pix, "
                f"file='{self.filepath}')")

    def get_pixel_scale_arcsec(self) -> float:
        """Get pixel scale in arcseconds per pixel."""
        return self.pixel_scale

    def world_to_pixel(self, ra: np.ndarray, dec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert world coordinates (RA, Dec) to pixel coordinates."""
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        # Create SkyCoord object
        coords = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')

        # Convert to pixel coordinates
        x_pix, y_pix = self.wcs.world_to_pixel(coords)

        return x_pix, y_pix

    def get_value_at_position(self, x: float, y: float) -> Tuple[float, float]:
        """Get data value and error at pixel position."""
        xi, yi = int(np.round(x)), int(np.round(y))

        if 0 <= xi < self.shape[1] and 0 <= yi < self.shape[0]:
            value = self.data[yi, xi]
            error = self.error[yi, xi]
            return value, error
        else:
            return np.nan, np.nan


def load_group_catalog(filepath: str, verbose: bool = True) -> GroupCatalog:
    """
    Load galaxy group catalog from FITS file.

    Parameters
    ----------
    filepath : str
        Path to the FITS catalog file
    verbose : bool, optional
        Print information about loaded catalog

    Returns
    -------
    GroupCatalog
        Container with group catalog data

    Raises
    ------
    FileNotFoundError
        If the catalog file does not exist
    ValueError
        If required columns are missing
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Catalog file not found: {filepath}")

    if verbose:
        logger.info(f"Loading group catalog from: {filepath}")

    # Load FITS file
    with fits.open(filepath) as hdul:
        # Get the table (usually extension 1)
        if len(hdul) > 1:
            data = Table(hdul[1].data)
        else:
            raise ValueError("No table extension found in FITS file")

    # Validate required columns (check for alternative names)
    has_ra = any(col in data.colnames for col in ['Ra', 'RA'])
    has_dec = any(col in data.colnames for col in ['Dec', 'DEC'])
    has_z = any(col in data.colnames for col in ['z', 'Redshift'])

    missing_cols = []
    if not has_ra:
        missing_cols.append('Ra/RA')
    if not has_dec:
        missing_cols.append('Dec/DEC')
    if not has_z:
        missing_cols.append('z/Redshift')

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Create catalog object
    catalog = GroupCatalog(data, str(filepath))

    if verbose:
        logger.info(f"Loaded {catalog.n_groups} groups")
        logger.info(f"Redshift range: z = [{np.min(catalog.get_redshifts()):.3f}, "
                   f"{np.max(catalog.get_redshifts()):.3f}]")

    return catalog


def load_xray_maps(map_filepath: str, error_filepath: str,
                   verbose: bool = True) -> XrayMap:
    """
    Load X-ray map and error map from FITS files.

    Parameters
    ----------
    map_filepath : str
        Path to the X-ray map FITS file
    error_filepath : str
        Path to the X-ray error map FITS file
    verbose : bool, optional
        Print information about loaded maps

    Returns
    -------
    XrayMap
        Container with X-ray map data, error, and WCS

    Raises
    ------
    FileNotFoundError
        If map files do not exist
    ValueError
        If map shapes don't match
    """
    map_path = Path(map_filepath)
    error_path = Path(error_filepath)

    if not map_path.exists():
        raise FileNotFoundError(f"X-ray map not found: {map_path}")
    if not error_path.exists():
        raise FileNotFoundError(f"X-ray error map not found: {error_path}")

    if verbose:
        logger.info(f"Loading X-ray map from: {map_path}")
        logger.info(f"Loading error map from: {error_path}")

    # Load X-ray map
    with fits.open(map_path) as hdul:
        data = hdul[0].data.astype(np.float64)
        header = hdul[0].header
        wcs = WCS(header)

    # Load error map
    with fits.open(error_path) as hdul:
        error = hdul[0].data.astype(np.float64)

    # Validate shapes match
    if data.shape != error.shape:
        raise ValueError(f"Map shapes don't match: {data.shape} vs {error.shape}")

    # Create map object
    xray_map = XrayMap(data, error, wcs, header, str(map_path))

    if verbose:
        logger.info(f"Map shape: {xray_map.shape}")
        logger.info(f"Pixel scale: {xray_map.pixel_scale:.3f} arcsec/pixel")
        logger.info(f"Data range: [{np.nanmin(data):.3e}, {np.nanmax(data):.3e}]")
        logger.info(f"Error range: [{np.nanmin(error):.3e}, {np.nanmax(error):.3e}]")

    return xray_map


def validate_coverage(catalog: GroupCatalog, xray_map: XrayMap) -> np.ndarray:
    """
    Check which groups have coverage in the X-ray map.

    Parameters
    ----------
    catalog : GroupCatalog
        Galaxy group catalog
    xray_map : XrayMap
        X-ray map with WCS

    Returns
    -------
    np.ndarray
        Boolean array indicating which groups have X-ray coverage
    """
    ra, dec = catalog.get_coordinates()
    x_pix, y_pix = xray_map.world_to_pixel(ra, dec)

    # Check if positions are within map boundaries
    in_bounds = (
        (x_pix >= 0) & (x_pix < xray_map.shape[1]) &
        (y_pix >= 0) & (y_pix < xray_map.shape[0])
    )

    # Check if data is not NaN at these positions
    has_data = np.zeros(len(ra), dtype=bool)
    for i, (x, y) in enumerate(zip(x_pix[in_bounds], y_pix[in_bounds])):
        xi, yi = int(np.round(x)), int(np.round(y))
        has_data[np.where(in_bounds)[0][i]] = not np.isnan(xray_map.data[yi, xi])

    coverage = in_bounds & has_data

    logger.info(f"Groups with X-ray coverage: {np.sum(coverage)}/{len(ra)} "
               f"({100*np.sum(coverage)/len(ra):.1f}%)")

    return coverage
