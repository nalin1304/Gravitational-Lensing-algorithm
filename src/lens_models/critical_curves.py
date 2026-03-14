"""
Critical Curves, Caustics, Magnification Maps, and Image Position Solver

Computes fundamental strong-lensing observables from any mass profile that
implements the ``MassProfile`` interface (deflection_angle, convergence).

Theory references
-----------------
- Critical curves & caustics:
    Schneider, Ehlers & Falco (1992), §5.4 — "Gravitational Lenses"
    Schneider, Kochanek & Wambsganss (2006), Ch. 3
- Magnification:
    Schneider (1992), Eq. 3.13–3.17:
        μ = 1 / det(A),  A_ij = δ_ij − ∂α_i/∂θ_j
    Equivalently: μ = 1 / [(1 − κ)² − γ²]
- Image position solver:
    Birrer & Amara (2018), §3.1 — Newton–Raphson with multiple initial seeds
    Kayser, Refsdal & Stabell (1986) — image classification (min/saddle/max)
"""

import numpy as np
from typing import Tuple, Optional, List, Dict
from scipy import optimize


# --------------------------------------------------------------------------- #
#  Jacobian and magnification utilities                                       #
# --------------------------------------------------------------------------- #

def lens_jacobian(
    profile,
    x: np.ndarray,
    y: np.ndarray,
    dx: float = 1e-5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the 2×2 lens-mapping Jacobian A_ij = δ_ij − ∂α_i/∂θ_j
    via centred finite differences on the deflection field.

    Parameters
    ----------
    profile : MassProfile
        Any object with ``deflection_angle(x, y) → (α_x, α_y)``.
    x, y : np.ndarray
        Image-plane coordinates (arcsec), any shape.
    dx : float
        Step size for finite differences (arcsec).

    Returns
    -------
    A11, A12, A21, A22 : np.ndarray
        Components of the Jacobian matrix A (same shape as *x*).

    Notes
    -----
    Schneider (1992) Eq. 3.13:  A_ij = δ_ij − ψ_{,ij}
    where ψ is the lensing potential and α_i = ψ_{,i}.
    """
    ax_px, ay_px = profile.deflection_angle(x + dx, y)
    ax_mx, ay_mx = profile.deflection_angle(x - dx, y)
    ax_py, ay_py = profile.deflection_angle(x, y + dx)
    ax_my, ay_my = profile.deflection_angle(x, y - dx)

    dax_dx = (ax_px - ax_mx) / (2.0 * dx)
    day_dx = (ay_px - ay_mx) / (2.0 * dx)
    dax_dy = (ax_py - ax_my) / (2.0 * dx)
    day_dy = (ay_py - ay_my) / (2.0 * dx)

    A11 = 1.0 - dax_dx
    A12 = -dax_dy
    A21 = -day_dx
    A22 = 1.0 - day_dy

    return A11, A12, A21, A22


def magnification_map(
    profile,
    x: np.ndarray,
    y: np.ndarray,
    dx: float = 1e-5,
) -> np.ndarray:
    """
    Compute the signed magnification μ = 1/det(A) over an image-plane grid.

    Parameters
    ----------
    profile : MassProfile
        Mass profile with ``deflection_angle`` method.
    x, y : np.ndarray
        Image-plane coordinate grids (arcsec).
    dx : float
        Finite-difference step (arcsec).

    Returns
    -------
    mu : np.ndarray
        Signed magnification at each grid point.

    Notes
    -----
    Schneider (1992) Eq. 3.17:
        μ = 1 / det(A) = 1 / [(1 − κ)² − γ²]
    where A is the lens Jacobian.  Values diverge on critical curves
    (det(A) → 0), so we clip to ±1000 for numerical safety.
    """
    A11, A12, A21, A22 = lens_jacobian(profile, x, y, dx)
    det_A = A11 * A22 - A12 * A21

    # Clip to avoid infinities on critical curves
    det_A = np.where(np.abs(det_A) < 1e-10, np.sign(det_A + 1e-30) * 1e-10, det_A)
    mu = 1.0 / det_A
    return np.clip(mu, -1000.0, 1000.0)


def convergence_shear(
    profile,
    x: np.ndarray,
    y: np.ndarray,
    dx: float = 1e-5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decompose the Jacobian into convergence κ and complex shear (γ₁, γ₂).

    Schneider (1992) Eq. 3.14–3.15:
        κ  = ½(ψ_{,11} + ψ_{,22})
        γ₁ = ½(ψ_{,11} − ψ_{,22})
        γ₂ = ψ_{,12}

    Returns
    -------
    kappa, gamma1, gamma2 : np.ndarray
    """
    A11, A12, A21, A22 = lens_jacobian(profile, x, y, dx)
    kappa = 1.0 - 0.5 * (A11 + A22)
    gamma1 = 0.5 * (A22 - A11)
    gamma2 = -0.5 * (A12 + A21)
    return kappa, gamma1, gamma2


# --------------------------------------------------------------------------- #
#  Critical curves and caustics                                               #
# --------------------------------------------------------------------------- #

def find_critical_curves(
    profile,
    grid_size: int = 200,
    grid_range: float = 3.0,
    dx: float = 1e-5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find critical curves (image plane) by locating det(A) = 0 contours.

    Uses marching-squares contour extraction on the determinant map.

    Parameters
    ----------
    profile : MassProfile
        Mass profile with ``deflection_angle`` method.
    grid_size : int
        Number of points per side of the evaluation grid.
    grid_range : float
        Half-width of the grid in arcseconds (grid spans [-range, +range]).
    dx : float
        Finite-difference step for Jacobian computation (arcsec).

    Returns
    -------
    crit_x, crit_y : np.ndarray
        Coordinates of points on the critical curves (arcsec).

    Notes
    -----
    Critical curves are the locus where det(A) = 0, i.e. where the
    magnification diverges (Schneider et al. 1992, §5.4).  For an SIS,
    the critical curve is the Einstein ring.  For NFW profiles, both
    tangential and radial critical curves may exist.
    """
    theta = np.linspace(-grid_range, grid_range, grid_size)
    xx, yy = np.meshgrid(theta, theta)

    A11, A12, A21, A22 = lens_jacobian(profile, xx, yy, dx)
    det_A = A11 * A22 - A12 * A21

    # Extract zero-contour via sign changes between adjacent cells
    crit_x, crit_y = _extract_zero_contour(det_A, xx, yy)
    return crit_x, crit_y


def find_caustics(
    profile,
    grid_size: int = 200,
    grid_range: float = 3.0,
    dx: float = 1e-5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find caustics by mapping critical curves to the source plane.

    The caustic is the source-plane image of the critical curve under
    the lens mapping β = θ − α(θ).

    Parameters
    ----------
    profile : MassProfile
        Mass profile with ``deflection_angle`` method.
    grid_size : int
        Number of grid points per side.
    grid_range : float
        Half-width of the image-plane grid (arcsec).
    dx : float
        Finite-difference step (arcsec).

    Returns
    -------
    caust_x, caust_y : np.ndarray
        Source-plane coordinates of the caustic (arcsec).

    Notes
    -----
    For a circularly symmetric lens, caustics are points (point caustic)
    or circles (tangential/radial caustics).  For elliptical lenses,
    caustics form the characteristic astroid/diamond shapes (Schneider 1992 §5.4).
    """
    crit_x, crit_y = find_critical_curves(profile, grid_size, grid_range, dx)

    if len(crit_x) == 0:
        return np.array([]), np.array([])

    ax, ay = profile.deflection_angle(crit_x, crit_y)
    caust_x = crit_x - ax
    caust_y = crit_y - ay
    return caust_x, caust_y


def tangential_and_radial_critical_curves(
    profile,
    grid_size: int = 200,
    grid_range: float = 3.0,
    dx: float = 1e-5,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Separate tangential and radial critical curves.

    The eigenvalues of the Jacobian A are (1-κ-γ) and (1-κ+γ).
    - Tangential critical curve: 1 − κ − γ = 0
    - Radial critical curve:     1 − κ + γ = 0

    Returns
    -------
    dict with keys 'tangential' and 'radial', each mapping to (x, y) arrays.

    References
    ----------
    Schneider, Ehlers & Falco (1992), Eq. 5.18–5.19.
    """
    theta = np.linspace(-grid_range, grid_range, grid_size)
    xx, yy = np.meshgrid(theta, theta)

    kappa, gamma1, gamma2 = convergence_shear(profile, xx, yy, dx)
    gamma = np.sqrt(gamma1**2 + gamma2**2)

    lambda_t = 1.0 - kappa - gamma  # tangential eigenvalue
    lambda_r = 1.0 - kappa + gamma  # radial eigenvalue

    tang_x, tang_y = _extract_zero_contour(lambda_t, xx, yy)
    rad_x, rad_y = _extract_zero_contour(lambda_r, xx, yy)

    return {
        'tangential': (tang_x, tang_y),
        'radial': (rad_x, rad_y),
    }


# --------------------------------------------------------------------------- #
#  Image position solver                                                      #
# --------------------------------------------------------------------------- #

def solve_lens_equation(
    profile,
    beta_x: float,
    beta_y: float,
    grid_size: int = 100,
    grid_range: float = 3.0,
    tol: float = 1e-8,
    max_iter: int = 50,
) -> List[Dict]:
    """
    Find all image positions for a given source position by solving β = θ − α(θ).

    Uses a two-phase approach:
    1. Coarse grid search to find candidate regions (sign changes in residuals)
    2. Newton–Raphson refinement from each candidate seed

    Parameters
    ----------
    profile : MassProfile
        Mass profile with ``deflection_angle`` method.
    beta_x, beta_y : float
        Source position in arcseconds.
    grid_size : int
        Coarse grid resolution per axis.
    grid_range : float
        Half-width of the search grid (arcsec).
    tol : float
        Convergence tolerance for Newton–Raphson.
    max_iter : int
        Maximum Newton–Raphson iterations per seed.

    Returns
    -------
    images : list of dict
        Each dict contains:
        - 'x', 'y': image position (arcsec)
        - 'magnification': signed magnification μ
        - 'parity': +1 (minimum/maximum) or −1 (saddle point)
        - 'type': 'minimum', 'saddle', or 'maximum' (Fermat surface classification)

    Notes
    -----
    The image classification follows Schneider (1992) §5.3:
    - Minima of τ: det(A) > 0, tr(A) > 0
    - Saddle points: det(A) < 0
    - Maxima of τ: det(A) > 0, tr(A) < 0

    For a point mass, there are exactly 2 images.  For SIS inside
    the Einstein radius, 2 images.  For NFW and elliptical profiles,
    typically 1, 3, or 5 images depending on source position relative
    to caustics (Kayser, Refsdal & Stabell 1986).
    """
    # Phase 1: coarse grid to find seed positions
    theta = np.linspace(-grid_range, grid_range, grid_size)
    xx, yy = np.meshgrid(theta, theta)

    ax, ay = profile.deflection_angle(xx, yy)
    residual_x = xx - ax - beta_x
    residual_y = yy - ay - beta_y
    residual_mag = residual_x**2 + residual_y**2

    # Find local minima of |β − (θ − α)|² as seed candidates
    seeds = _find_residual_minima(residual_mag, xx, yy, threshold_factor=5.0)

    # Phase 2: Newton–Raphson refinement
    images = []
    found_positions = []

    for sx, sy in seeds:
        result = _newton_raphson_lens(
            profile, beta_x, beta_y, sx, sy, tol, max_iter
        )
        if result is not None:
            # Check for duplicates
            is_duplicate = False
            for fx, fy in found_positions:
                if np.sqrt((result[0] - fx)**2 + (result[1] - fy)**2) < 10 * tol:
                    is_duplicate = True
                    break
            if not is_duplicate:
                found_positions.append((result[0], result[1]))
                img_info = _classify_image(profile, result[0], result[1])
                images.append(img_info)

    # Sort by magnification (brightest first)
    images.sort(key=lambda im: -abs(im['magnification']))
    return images


# --------------------------------------------------------------------------- #
#  Full analysis (convenience wrapper)                                        #
# --------------------------------------------------------------------------- #

def full_lensing_analysis(
    profile,
    grid_size: int = 200,
    grid_range: float = 3.0,
    source_positions: Optional[List[Tuple[float, float]]] = None,
) -> Dict:
    """
    Perform a complete strong-lensing analysis for a given mass profile.

    Returns critical curves, caustics, magnification map, and optionally
    solves the lens equation for specified source positions.

    Parameters
    ----------
    profile : MassProfile
        Mass profile to analyse.
    grid_size : int
        Grid resolution per axis.
    grid_range : float
        Half-width of analysis grid (arcsec).
    source_positions : list of (beta_x, beta_y), optional
        Source positions for image finding.

    Returns
    -------
    dict with keys:
        'critical_curves': (x, y) arrays
        'caustics': (x, y) arrays
        'magnification_map': 2D array
        'grid_x', 'grid_y': coordinate arrays
        'images': list of image solutions (if source_positions given)
    """
    theta = np.linspace(-grid_range, grid_range, grid_size)
    xx, yy = np.meshgrid(theta, theta)

    mu = magnification_map(profile, xx, yy)
    crit_x, crit_y = find_critical_curves(profile, grid_size, grid_range)
    caust_x, caust_y = find_caustics(profile, grid_size, grid_range)

    result = {
        'critical_curves': (crit_x, crit_y),
        'caustics': (caust_x, caust_y),
        'magnification_map': mu,
        'grid_x': xx,
        'grid_y': yy,
        'grid_extent': [-grid_range, grid_range, -grid_range, grid_range],
    }

    if source_positions:
        all_images = []
        for bx, by in source_positions:
            imgs = solve_lens_equation(
                profile, bx, by, grid_size=grid_size, grid_range=grid_range
            )
            all_images.append({
                'source': (bx, by),
                'images': imgs,
                'n_images': len(imgs),
            })
        result['image_solutions'] = all_images

    return result


# --------------------------------------------------------------------------- #
#  Internal helpers                                                           #
# --------------------------------------------------------------------------- #

def _extract_zero_contour(
    field: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract zero-level contour points from a 2D scalar field using
    marching-squares sign-change detection with linear interpolation.
    """
    contour_x = []
    contour_y = []
    ny, nx = field.shape

    for i in range(ny - 1):
        for j in range(nx - 1):
            # Check horizontal edge (i, j) → (i, j+1)
            if field[i, j] * field[i, j + 1] < 0:
                frac = field[i, j] / (field[i, j] - field[i, j + 1])
                cx = xx[i, j] + frac * (xx[i, j + 1] - xx[i, j])
                cy = yy[i, j] + frac * (yy[i, j + 1] - yy[i, j])
                contour_x.append(cx)
                contour_y.append(cy)

            # Check vertical edge (i, j) → (i+1, j)
            if field[i, j] * field[i + 1, j] < 0:
                frac = field[i, j] / (field[i, j] - field[i + 1, j])
                cx = xx[i, j] + frac * (xx[i + 1, j] - xx[i, j])
                cy = yy[i, j] + frac * (yy[i + 1, j] - yy[i, j])
                contour_x.append(cx)
                contour_y.append(cy)

    return np.array(contour_x), np.array(contour_y)


def _find_residual_minima(
    residual_mag: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    threshold_factor: float = 5.0,
) -> List[Tuple[float, float]]:
    """
    Find seed positions where the lens equation residual is locally minimal.
    Uses a 3×3 neighbourhood comparison and adaptive thresholding.
    """
    ny, nx = residual_mag.shape
    step = (xx[0, 1] - xx[0, 0])
    threshold = threshold_factor * step**2

    seeds = []
    for i in range(1, ny - 1):
        for j in range(1, nx - 1):
            val = residual_mag[i, j]
            if val > threshold:
                continue
            neighbourhood = residual_mag[i - 1:i + 2, j - 1:j + 2]
            if val <= neighbourhood.min():
                seeds.append((xx[i, j], yy[i, j]))

    return seeds


def _newton_raphson_lens(
    profile,
    beta_x: float,
    beta_y: float,
    x0: float,
    y0: float,
    tol: float = 1e-8,
    max_iter: int = 50,
    dx: float = 1e-6,
) -> Optional[Tuple[float, float]]:
    """
    Refine an image position via Newton–Raphson iteration.

    Solves f(θ) = θ − α(θ) − β = 0  using the Jacobian A = I − ∂α/∂θ.
    """
    x, y = x0, y0

    for _ in range(max_iter):
        ax, ay = profile.deflection_angle(
            np.array([x]), np.array([y])
        )
        ax, ay = np.asarray(ax).item(), np.asarray(ay).item()

        fx = x - ax - beta_x
        fy = y - ay - beta_y

        if fx**2 + fy**2 < tol**2:
            return (x, y)

        # Jacobian via central differences (O(dx²) accuracy)
        ax_px, ay_px = profile.deflection_angle(
            np.array([x + dx]), np.array([y])
        )
        ax_mx, ay_mx = profile.deflection_angle(
            np.array([x - dx]), np.array([y])
        )
        ax_py, ay_py = profile.deflection_angle(
            np.array([x]), np.array([y + dx])
        )
        ax_my, ay_my = profile.deflection_angle(
            np.array([x]), np.array([y - dx])
        )
        ax_px = np.asarray(ax_px).item()
        ax_mx = np.asarray(ax_mx).item()
        ay_px = np.asarray(ay_px).item()
        ay_mx = np.asarray(ay_mx).item()
        ax_py = np.asarray(ax_py).item()
        ax_my = np.asarray(ax_my).item()
        ay_py = np.asarray(ay_py).item()
        ay_my = np.asarray(ay_my).item()

        ddx = 2.0 * dx
        A11 = 1.0 - (ax_px - ax_mx) / ddx
        A12 = -(ax_py - ax_my) / ddx
        A21 = -(ay_px - ay_mx) / ddx
        A22 = 1.0 - (ay_py - ay_my) / ddx

        det = A11 * A22 - A12 * A21
        if abs(det) < 1e-15:
            return None

        # Newton step: θ_new = θ - A⁻¹ f
        x -= (A22 * fx - A12 * fy) / det
        y -= (-A21 * fx + A11 * fy) / det

    return None


def _classify_image(
    profile,
    x: float,
    y: float,
    dx: float = 1e-5,
) -> Dict:
    """
    Classify an image as minimum, saddle, or maximum of the Fermat potential.

    Schneider (1992) §5.3:
    - det(A) > 0, tr(A) > 0  →  minimum (type I)
    - det(A) < 0             →  saddle  (type II)
    - det(A) > 0, tr(A) < 0  →  maximum (type III)
    """
    x_arr = np.array([x])
    y_arr = np.array([y])

    A11, A12, A21, A22 = lens_jacobian(profile, x_arr, y_arr, dx)
    det_A = np.asarray(A11 * A22 - A12 * A21).item()
    tr_A = np.asarray(A11 + A22).item()

    # Magnification
    if abs(det_A) < 1e-12:
        mu = 1000.0 if det_A >= 0 else -1000.0
    else:
        mu = 1.0 / det_A

    parity = 1 if det_A > 0 else -1

    if det_A > 0 and tr_A > 0:
        img_type = 'minimum'
    elif det_A < 0:
        img_type = 'saddle'
    else:
        img_type = 'maximum'

    return {
        'x': x,
        'y': y,
        'magnification': mu,
        'parity': parity,
        'type': img_type,
        'det_A': det_A,
        'tr_A': tr_A,
    }

