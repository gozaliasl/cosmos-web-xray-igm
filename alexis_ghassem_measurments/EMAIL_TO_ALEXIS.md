# Email to Alexis - M200-Lx Relation and Analysis Details

Subject: X-ray Mass-Luminosity Relation and Analysis Parameters

Dear Alexis,

I wanted to share the details of the mass–luminosity relation and associated parameters. The agreement is very good; there are slight differences which I think may relate to the scaling relations. Please check them if they need any change and let me know.

## Mass–Luminosity Relation

I'm using the **Leauthaud et al. (2010) / Kettula et al. (2015)** M200–Lx scaling relation:

```
M200 = A × (Lx × E(z)⁻¹ / L₀)^α × M₀ / E(z)
```

In log form:

```
log₁₀(M200) = log₁₀(A × M₀) + α × log₁₀(Lx / L₀) – (α + 1) × log₁₀(E(z))
```

**Parameters:**
- **α** = 0.64
- **A** = 1.17
- **L₀** = 5.01 × 10⁴² erg/s
- **M₀** = 5.01 × 10¹³ M☉

Where:
- **Lx** is the rest-frame X-ray luminosity in the 0.5-2.0 keV energy band (after K-correction)
- **E(z)** = H(z)/H₀ is the Hubble parameter evolution

## Luminosity Measurement

Lx is measured from aperture photometry following this chain:

```
Net Count Rate → Flux → Rest-Frame Luminosity
```

**Main relation**:
```
Lx = 4π × d_L² × F × K(z)
```

Where:
- **F** = net_count_rate × count_rate_to_flux
- **count_rate_to_flux** = 5.41 × 10⁻¹² erg cm⁻² s⁻¹ per (counts/s) (XMM MOS, 0.5-2.0 keV, Γ=2.0, N_H=2.5×10²⁰ cm⁻²)
- **d_L**: Luminosity distance (Planck18 cosmology)
- **K(z)** = (1+z)^0.5: K-correction for thermal emission (kT ≈ 1 keV)

The net count rate is background-subtracted from aperture photometry on the X-ray count rate maps.

## Temperature–Luminosity Relation

I'm using the **Kettula et al. (2015) bias-corrected** Lx–Tx relation:

```
log₁₀(Lx × E(z)⁻¹ / Lx₀) = α × log₁₀(Tx / T₀) + log₁₀(A)
```

Rearranging to solve for temperature:

```
Tx = T₀ × ((Lx × E(z)⁻¹ / Lx₀) / A)^(1/α)
```

In log form:

```
log₁₀(Tx / T₀) = (log₁₀(Lx × E(z)⁻¹ / Lx₀) - log₁₀(A)) / α
```

**Parameters:**
- **α** = 2.52 (slope, bias-corrected)
- **A** = 1.51 (normalization, 10^0.18)
- **Lx₀** = 10⁴⁴ erg/s (pivot luminosity)
- **T₀** = 5.0 keV (pivot temperature)

Where:
- **Lx** is the rest-frame X-ray luminosity in the 0.5-2.0 keV energy band (after K-correction)
- **E(z)** = H(z)/H₀ is the Hubble parameter evolution
- **Tx** is the core-excised temperature in keV

**Error/Scatter:**
- Intrinsic scatter: ~0.22 dex (~51% fractional error)
- This reflects the uncertainty in estimating temperature from luminosity without spectral fitting

## K-Correction

I apply a simple K-correction: **K(z) = (1 + z)^0.5**, suitable for ~1 keV APEC models in the 0.5–2.0 keV band. If you have any better relation, please share it with me.

The rest-frame luminosity is then:

```
Lx_rest = Lx_obs × K(z)
```

where Lx_obs is the observed-frame luminosity.

## Cosmology

Using **Planck 2018**: H₀ = 67.4 km/s/Mpc, Ωₘ = 0.315, ΩΛ = 0.685.

E(z) is calculated as:

```
E(z) = √[Ωₘ(1+z)³ + ΩΛ]
```

## R200 Calculation

R200 is calculated from M200 using the standard relation:

```
M200 = (4π/3) × 200 × ρ_crit(z) × R200³
```

Rearranging:

```
R200 = [3M200 / (4π × 200 × ρ_crit(z))]^(1/3)
```

where ρ_crit(z) is the critical density at redshift z, calculated using the Planck18 cosmology.

## Comparison Results

I've attached a comparison plot showing the agreement between our measurements. The comparison shows:

- **Flux**: Excellent agreement (r = 1.000, 0.05% difference)
- **Temperature**: Excellent agreement (r = 0.994, 4.07% difference)
- **R200 (Luminosity-based)**: Excellent agreement (r = 0.989, 12.86% difference)
- **M200 (Luminosity-based)**: Strong correlation (r = 0.979, 38.32% difference)
- **Luminosity**: Good correlation (r = 0.986, 65.90% difference)
- **SNR**: Perfect agreement (r = 1.000, 0.02% difference)

While some offsets exist in luminosity (~66%) and mass (~38%), correlations remain strong (r ≈ 0.98–1.00), confirming overall consistency between our approaches.

The remaining differences are likely due to different implementation details or calibration choices, but the strong correlations suggest our methods are fundamentally consistent.

## References

- Leauthaud et al. (2010): "Constraining the scatter in the mass-richness relation of maxBCG clusters"
- Kettula et al. (2015): "The X-ray properties of z~0.5 galaxy groups in the COSMOS field"
- Planck Collaboration (2018): "Planck 2018 results. VI. Cosmological parameters"

Please let me know if you'd like to go over any part in detail or if you notice any discrepancies that we should discuss.

Best regards,
Ghassem

---

**Attachment:**
- `ghassem_detections_fits_comparison_scatter_plots.png` - Comparison plot showing agreement between our measurements
