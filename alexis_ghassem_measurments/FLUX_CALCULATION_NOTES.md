# Flux Calculation: Count Rate Usage

## Summary

The pipeline uses **Net Count Rate** (background-subtracted count rate in counts/s) to determine flux.

## Details

### Count Rate Source

1. **Input**: The pipeline receives count rate maps (already in units of counts/s per pixel)
2. **Photometry**: Aperture photometry is performed to extract:
   - `source_counts`: Total counts in aperture (counts/s)
   - `background`: Background level (counts/s)
   - `net_counts`: Background-subtracted count rate = `source_counts - background_scaled` (counts/s)

3. **Flux Calculation**: The `calculate_xray_flux()` function uses:
   ```python
   flux = net_counts * count_rate_to_flux
   ```
   where `net_counts` is already in count rate units (counts/s), and `exposure_time=None` (so no conversion is needed).

### Code Reference

**Location**: `src/xray_analysis/xray_properties.py`

```python
def calculate_xray_flux(
    net_counts: np.ndarray,  # Already in count rate (counts/s)
    net_error: np.ndarray,
    count_rate_to_flux: float = 2.6e-12,
    exposure_time: Optional[float] = None,  # None = assumes counts are already count rates
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    if exposure_time is not None:
        # Convert counts to count rate
        count_rate = net_counts / exposure_time
    else:
        count_rate = net_counts  # Already count rate
    
    # Convert count rate to flux
    flux = count_rate * count_rate_to_flux
    return flux, flux_error
```

**Photometry**: `src/xray_analysis/photometry.py`
```python
# Calculate net counts (background-subtracted)
net_counts = source_counts - background_scaled  # Both in count rate units
```

### Conversion Factor

The `count_rate_to_flux` conversion factor is set in the configuration:

- **config.yaml**: `1.082e-11` erg cm⁻² s⁻¹ per (counts/s)
  - Comment: "5.41e-12 each MOS detector *2 for 2 detector MOS count-rate -> flux"
  - For: MOS1+MOS2, 0.5-2 keV, Gamma=2.0, NH=2.5e20 cm⁻² (COSMOS field)

- **config.txt**: `2.6e-12` erg cm⁻² s⁻¹ per (counts/s)
  - Comment: "MOS count-rate -> flux (counts/s to erg/cm^2/s)"
  - For: XMM MOS1+MOS2, 0.5-2 keV, kT≈1 keV APEC spectrum

### Ghassem's Data Structure

From `alexis_ghassem.csv`, Ghassem's measurements include:
- `Source_Counts`: Raw source count rate (counts/s) before background subtraction
- `Net_Counts`: Background-subtracted count rate (counts/s)
- `Flux_erg_cm2_s`: Calculated flux using Net_Counts

**Example** (first row):
- Source_Counts = 1.240e-4 counts/s
- Background = -3.468e-6 counts/s (negative suggests background subtraction already applied)
- Net_Counts = 1.275e-4 counts/s
- Flux_erg_cm2_s = 1.379e-15 erg cm⁻² s⁻¹

### Key Point

**The pipeline uses Net_Counts (net count rate) for flux calculation, NOT Source_Counts.**

This means:
- Flux = Net_Counts × count_rate_to_flux
- Net_Counts = Source_Counts - Background (scaled to aperture area)

### Implications for the 2× Flux Discrepancy

The 2× factor (slope = 1.9989) between Alexis and Ghassem's flux measurements could be due to:

1. **Different count_rate_to_flux conversion factors**:
   - If Alexis uses 2.6e-12 and Ghassem uses ~5.2e-12, that would explain the 2× factor
   - Or if one uses single detector and the other uses combined detectors

2. **Different background subtraction methods**:
   - Different background estimation could affect Net_Counts
   - Different aperture sizes would scale background differently

3. **Different energy bands or spectral assumptions**:
   - The conversion factor depends on spectral model (APEC temperature, NH, etc.)
   - Different assumptions → different conversion factors

4. **Unit confusion**:
   - One might be using counts (needs exposure time) while the other uses count rate
   - But this seems unlikely given the perfect correlation

### Recommendation

To resolve the 2× discrepancy, check:
1. What `count_rate_to_flux` value was used by each analyst
2. Whether both used Net_Counts or if one used Source_Counts
3. The spectral model assumptions (temperature, NH, energy band) used for the conversion factor
4. Whether the count rate maps are in the same units (single detector vs. combined detectors)

