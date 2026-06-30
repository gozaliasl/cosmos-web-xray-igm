import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
z = 0.71

# Calculate angular diameter distance
da = cosmo.angular_diameter_distance(z).to(u.Mpc).value

# Convert 750 kpc physical to angular
r_physical = 0.75  # Mpc
theta_rad = r_physical / da
theta_arcmin = np.degrees(theta_rad) * 60

print(f'At z={z}:')
print(f'  Angular diameter distance: {da:.2f} Mpc')
print(f'  750 kpc physical radius -> {theta_arcmin:.2f} arcmin')
print()
print(f'Current search: 5 arcmin -> {5 * da / 60 * 1000:.0f} kpc (TOO BIG!)')
print(f'Recommended: {theta_arcmin:.2f} arcmin -> 750 kpc')
