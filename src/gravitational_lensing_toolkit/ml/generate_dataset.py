import numpy as np

def generate_convergence_map_vectorized(lens_model, grid_size: int = 64, extent: float = 3.0) -> np.ndarray:
	"""Vectorized convergence map generator compatible with the toolkit namespace.

	This mirrors the original implementation but avoids importing the legacy
	module that relies on relative imports.
	"""
	# Create coordinate grid
	x = np.linspace(-extent, extent, grid_size)
	y = np.linspace(-extent, extent, grid_size)
	X, Y = np.meshgrid(x, y)

	# Flatten grid for vectorized computation
	x_flat = X.ravel()
	y_flat = Y.ravel()

	# Compute convergence at all points simultaneously
	kappa_flat = lens_model.convergence(x_flat, y_flat)

	# Reshape back to 2D grid
	return kappa_flat.reshape(grid_size, grid_size)


def generate_synthetic_convergence(
	*,
	profile_type: str,
	mass: float,
	scale_radius: float,
	ellipticity: float,
	grid_size: int
):
	"""Generate a synthetic convergence map and coordinate grids.

	Returns (convergence_map, X, Y) to match API expectations.
	"""
	# Lazy imports of lensing pieces via toolkit namespace
	from gravitational_lensing_toolkit.lens_models.lens_system import LensSystem
	from gravitational_lensing_toolkit.lens_models.mass_profiles import NFWProfile
	try:
		from gravitational_lensing_toolkit.lens_models.advanced_profiles import EllipticalNFWProfile
	except Exception:
		EllipticalNFWProfile = None  # type: ignore

	# Simple cosmology defaults
	lens = LensSystem(z_lens=0.5, z_source=2.0)

	if profile_type == "NFW" or EllipticalNFWProfile is None:
		model = NFWProfile(M_vir=mass, concentration=scale_radius / 20.0 if scale_radius > 0 else 5.0, lens_system=lens)
	else:
		model = EllipticalNFWProfile(M_vir=mass, c=scale_radius / 20.0 if scale_radius > 0 else 5.0, lens_sys=lens, ellipticity=ellipticity)

	# Coordinates (arcsec extent)
	extent = 3.0
	x = np.linspace(-extent, extent, grid_size)
	y = np.linspace(-extent, extent, grid_size)
	X, Y = np.meshgrid(x, y)

	conv = generate_convergence_map_vectorized(model, grid_size=grid_size, extent=extent)
	return conv, X, Y
