"""
MAST Archive Downloader for SLACS Lens HST/ACS Images

Automated download of real Hubble Space Telescope observations from the
Mikulski Archive for Space Telescopes (MAST) using the astroquery API.

Targets SLACS Survey lenses (Bolton et al. 2008, ApJ 682, 964):
  - HST Proposal IDs: 10886, 10494, 10174
  - Instrument: ACS/WFC
  - Primary filter: F814W (I-band)

Usage:
    from src.data.mast_downloader import MASTDownloader
    dl = MASTDownloader(cache_dir="data/hst_cache")
    fits_path = dl.download_slacs_lens("SDSS J0946+1006")

Author: Gravitational Lensing Research Platform
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypedDict
import hashlib

import numpy as np
import logging

logger = logging.getLogger(__name__)

# Astroquery is optional — required for live MAST queries
try:
    from astroquery.mast import Observations
    ASTROQUERY_AVAILABLE = True
except ImportError:
    ASTROQUERY_AVAILABLE = False

try:
    from astropy.io import fits
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False


# ================================================================
# SLACS Survey Catalog with HST Proposal IDs and coordinates
# (Bolton et al. 2008, ApJ 682, 964; Auger et al. 2010, ApJ 724, 511)
# ================================================================
class LensCatalogEntry(TypedDict):
    name: str
    ra: float
    dec: float
    z_lens: float
    z_source: float
    sigma_v: float
    einstein_radius: float
    proposal_ids: List[int]
    filter: str
    ref: str


SLACS_HST_CATALOG: List[LensCatalogEntry] = [
    {
        "name": "SDSS J0946+1006",
        "ra": 146.59917,  "dec": 10.10806,
        "z_lens": 0.222,  "z_source": 0.609,
        "sigma_v": 263.0,  "einstein_radius": 1.38,
        "proposal_ids": [10886],
        "filter": "F814W",
        "ref": "Bolton+2008",
    },
    {
        "name": "SDSS J1250+0523",
        "ra": 192.58167,  "dec": 5.39528,
        "z_lens": 0.232,  "z_source": 0.795,
        "sigma_v": 252.0,  "einstein_radius": 1.13,
        "proposal_ids": [10886],
        "filter": "F814W",
        "ref": "Bolton+2008",
    },
    {
        "name": "SDSS J1402+6321",
        "ra": 210.63375,  "dec": 63.35806,
        "z_lens": 0.205,  "z_source": 0.481,
        "sigma_v": 267.0,  "einstein_radius": 1.35,
        "proposal_ids": [10886],
        "filter": "F814W",
        "ref": "Bolton+2008",
    },
    {
        "name": "SDSS J0252+0039",
        "ra": 43.01500,  "dec": 0.65556,
        "z_lens": 0.280,  "z_source": 0.982,
        "sigma_v": 164.0,  "einstein_radius": 1.04,
        "proposal_ids": [10886],
        "filter": "F814W",
        "ref": "Bolton+2008",
    },
    {
        "name": "SDSS J0037-0942",
        "ra": 9.34208,  "dec": -9.70833,
        "z_lens": 0.195,  "z_source": 0.632,
        "sigma_v": 279.0,  "einstein_radius": 1.53,
        "proposal_ids": [10494],
        "filter": "F814W",
        "ref": "Bolton+2008",
    },
    {
        "name": "SDSS J0737+3216",
        "ra": 114.36854,  "dec": 32.27181,
        "z_lens": 0.3223,  "z_source": 0.5812,
        "sigma_v": 322.0,  "einstein_radius": 1.00,
        "proposal_ids": [10174, 10494],
        "filter": "F814W",
        "ref": "Bolton+2008",
    },
    {
        "name": "SDSS J1205+4910",
        "ra": 181.41846,  "dec": 49.17481,
        "z_lens": 0.2150,  "z_source": 0.4808,
        "sigma_v": 281.0,  "einstein_radius": 1.22,
        "proposal_ids": [10174, 10494],
        "filter": "F814W",
        "ref": "Bolton+2008",
    },
    {
        "name": "SDSS J1630+4520",
        "ra": 247.61729,  "dec": 45.34339,
        "z_lens": 0.2479,  "z_source": 0.7933,
        "sigma_v": 279.0,  "einstein_radius": 1.81,
        "proposal_ids": [10174, 10494],
        "filter": "F814W",
        "ref": "Bolton+2008",
    },
    {
        "name": "SDSS J2321-0939",
        "ra": 350.33721,  "dec": -9.65285,
        "z_lens": 0.0819,  "z_source": 0.5324,
        "sigma_v": 245.0,  "einstein_radius": 1.57,
        "proposal_ids": [10174],
        "filter": "F814W",
        "ref": "Bolton+2008",
    },
]


class MASTDownloader:
    """
    Download real HST/ACS observations from the MAST archive.

    Implements a three-tier data strategy:
      1. Check local cache (fast)
      2. Query MAST by coordinates (requires astroquery + network)
      3. Generate synthetic observation matching published parameters (explicit opt-in only)
    """

    def __init__(self, cache_dir: str = "data/hst_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.catalog: Dict[str, LensCatalogEntry] = {
            entry["name"]: entry for entry in SLACS_HST_CATALOG
        }

    def list_available_lenses(self) -> List[str]:
        """List SLACS lenses in the catalog."""
        return list(self.catalog.keys())

    def download_slacs_lens(
        self,
        name: str,
        force: bool = False,
        search_radius_arcsec: float = 5.0,
        allow_synthetic_fallback: bool = False,
    ) -> Path:
        """
        Download a SLACS lens HST/ACS image.

        Parameters
        ----------
        name : str
            Lens name (e.g., "SDSS J0946+1006").
        force : bool
            Force re-download even if cached.
        search_radius_arcsec : float
            MAST coordinate search radius in arcseconds.
        allow_synthetic_fallback : bool
            If ``True``, generate a synthetic FITS surrogate when no archival
            observation can be fetched. This should remain ``False`` for
            publication-grade observational validation.

        Returns
        -------
        fits_path : Path
            Path to the downloaded/cached FITS file.

        Raises
        ------
        FileNotFoundError
            If no cached or archival observation is available and synthetic
            fallback is disabled.
        """
        if name not in self.catalog:
            raise ValueError(
                f"Unknown lens '{name}'. Available: {list(self.catalog.keys())}"
            )

        entry = self.catalog[name]
        safe_name = name.replace(" ", "_").replace("+", "p").replace("-", "m")
        cached_fits = self.cache_dir / f"{safe_name}_F814W_drz.fits"

        # Tier 1: Check cache
        if not force and cached_fits.exists():
            logger.info(f"Using cached FITS: {cached_fits}")
            return cached_fits

        # Tier 2: Query MAST
        if ASTROQUERY_AVAILABLE and ASTROPY_AVAILABLE:
            logger.info(f"Querying MAST for {name} at RA={entry['ra']}, Dec={entry['dec']}...")
            try:
                fits_path = self._query_and_download(entry, cached_fits, search_radius_arcsec)
                if fits_path is not None:
                    return fits_path
            except Exception as e:
                logger.warning(f"MAST query failed for {name}: {e}")

        if not allow_synthetic_fallback:
            raise FileNotFoundError(
                f"No cached or archival HST observation available for {name}. "
                "Enable allow_synthetic_fallback only for demo/smoke-test workflows."
            )

        # Tier 3: Generate synthetic with published parameters
        logger.info(f"Generating synthetic observation for {name} from published parameters")
        self._generate_synthetic_fits(entry, cached_fits)
        return cached_fits

    def _query_and_download(
        self,
        entry: LensCatalogEntry,
        output_path: Path,
        search_radius_arcsec: float,
    ) -> Optional[Path]:
        """Query MAST by coordinates and download drizzled ACS image."""
        coord = SkyCoord(
            ra=entry["ra"], dec=entry["dec"], unit=(u.deg, u.deg), frame="icrs"
        )

        # Query observations near the lens position
        obs_table = Observations.query_criteria(
            coordinates=coord,
            radius=search_radius_arcsec * u.arcsec,
            obs_collection="HST",
            instrument_name="ACS/WFC",
            filters=entry["filter"],
            dataproduct_type="image",
        )

        if len(obs_table) == 0:
            logger.warning(f"No MAST results for {entry['name']}")
            return None

        # Prefer observations from known SLACS proposal IDs
        preferred_mask = np.isin(
            obs_table["proposal_id"].astype(str),
            [str(pid) for pid in entry["proposal_ids"]],
        )
        if np.any(preferred_mask):
            obs_table = obs_table[preferred_mask]

        # Sort by exposure time (prefer deeper exposures)
        obs_table.sort("t_exptime")
        obs_table.reverse()

        # Get data products for the best observation
        best_obs = obs_table[0:1]
        products = Observations.get_product_list(best_obs)

        # Filter for drizzled science images (_drz.fits or _drc.fits)
        drz_mask = np.array([
            ("drz" in str(row["productFilename"]).lower() or
             "drc" in str(row["productFilename"]).lower())
            and row["productType"] == "SCIENCE"
            for row in products
        ])

        if not np.any(drz_mask):
            # Fall back to any calibrated science product
            drz_mask = np.array([
                row["productType"] == "SCIENCE"
                and row["calib_level"] >= 2
                for row in products
            ])

        if not np.any(drz_mask):
            logger.warning(f"No drizzled products found for {entry['name']}")
            return None

        filtered_products = products[drz_mask]

        # Download
        manifest = Observations.download_products(
            filtered_products[0:1],
            download_dir=str(self.cache_dir / "mast_raw"),
        )

        if len(manifest) > 0 and manifest["Status"][0] == "COMPLETE":
            downloaded_path = Path(manifest["Local Path"][0])
            # Move to standardized location
            import shutil
            shutil.move(str(downloaded_path), str(output_path))
            logger.info(f"Downloaded: {output_path}")
            return output_path

        return None

    def _generate_synthetic_fits(self, entry: LensCatalogEntry, output_path: Path):
        """
        Generate a synthetic FITS file matching SLACS parameters.

        Uses the project's NFW convergence map generator with parameters
        derived from the published velocity dispersion.
        """
        import sys
        project_root = Path(__file__).resolve().parent.parent.parent
        sys.path.insert(0, str(project_root))

        from src.lens_models.mass_profiles import NFWProfile
        from src.lens_models.lens_system import LensSystem
        from src.ml.generate_dataset import generate_convergence_map_vectorized

        # Derive virial mass from velocity dispersion
        sigma_v = entry["sigma_v"]
        M_vir = 1e12 * (sigma_v / 200.0) ** 4

        lens_sys = LensSystem(z_lens=entry["z_lens"], z_source=entry["z_source"])
        lens = NFWProfile(M_vir=M_vir, concentration=10.0, lens_system=lens_sys)

        grid_size = 128  # Higher resolution for "real data"
        extent = max(2.0 * entry["einstein_radius"], 2.0)

        kappa = generate_convergence_map_vectorized(
            lens_model=lens, grid_size=grid_size, extent=extent
        )

        # Add realistic HST-like noise
        seed_digest = hashlib.sha256(entry["name"].encode("utf-8")).digest()
        seed = int.from_bytes(seed_digest[:8], byteorder="big", signed=False) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        read_noise = rng.normal(0, 0.001, kappa.shape)   # Read noise
        sky_bg = rng.poisson(0.5, kappa.shape) * 0.001    # Sky background
        poisson_noise = rng.poisson(np.maximum(kappa * 100, 0)) / 100 - kappa  # Photon noise
        noisy_kappa = np.maximum(kappa + read_noise + sky_bg * 0.1 + poisson_noise * 0.05, 0)

        # Variance = read_noise_variance + sky_bg_variance + poisson_noise_variance
        # Note: Var(P(lambda)/c) = lambda / c^2. Here lambda = kappa*100, c = 100, factor = 0.05
        # Var = (0.05^2 * kappa*100) / 10000 = 0.000025 * kappa
        variance = 0.001**2 + (0.5 * 0.001**2) * 0.1**2 + np.maximum(kappa, 0) * 0.000025
        wht_kappa = 1.0 / (variance + 1e-10)

        # Write FITS
        if ASTROPY_AVAILABLE:
            hdr = fits.Header()
            hdr["TELESCOP"] = "HST"
            hdr["INSTRUME"] = "ACS"
            hdr["FILTER1"] = entry["filter"]
            hdr["RA_TARG"] = entry["ra"]
            hdr["DEC_TARG"] = entry["dec"]
            hdr["EXPTIME"] = 2400.0  # Typical SLACS exposure
            hdr["CD1_1"] = -(2 * extent / grid_size) / 3600  # Pixel scale → degrees
            hdr["CD2_2"] = (2 * extent / grid_size) / 3600
            hdr["CD1_2"] = 0.0
            hdr["CD2_1"] = 0.0
            hdr["CRPIX1"] = grid_size / 2
            hdr["CRPIX2"] = grid_size / 2
            hdr["CRVAL1"] = entry["ra"]
            hdr["CRVAL2"] = entry["dec"]
            hdr["CTYPE1"] = "RA---TAN"
            hdr["CTYPE2"] = "DEC--TAN"
            hdr["BUNIT"] = "ELECTRONS/S"
            hdr["OBJECT"] = entry["name"]
            hdr["PROPOSID"] = entry["proposal_ids"][0]
            hdr["COMMENT"] = "Synthetic observation from SLACS published parameters"
            hdr["COMMENT"] = f"sigma_v={sigma_v} km/s, M_vir={M_vir:.2e} Msun"
            hdr["COMMENT"] = f"z_lens={entry['z_lens']}, z_source={entry['z_source']}"

            hdul = fits.HDUList([
                fits.PrimaryHDU(header=hdr),
                fits.ImageHDU(noisy_kappa.astype(np.float32), name="SCI"),
                fits.ImageHDU(wht_kappa.astype(np.float32), name="WHT"),
            ])
            hdul.writeto(str(output_path), overwrite=True)
            logger.info(f"Synthetic FITS written: {output_path}")
        else:
            # Fallback: save as .npy
            npy_path = output_path.with_suffix(".npy")
            np.save(npy_path, noisy_kappa)
            wht_path = output_path.with_name(f"{output_path.stem}_wht.npy")
            np.save(wht_path, wht_kappa)
            logger.info(f"Synthetic NPY written: {npy_path}")

    def load_image(
        self,
        name: str,
        grid_size: int = 64,
        cutout_arcsec: Optional[float] = None,
        allow_synthetic_fallback: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Download (if needed) and load a SLACS lens image and weight map.

        Parameters
        ----------
        name : str
            Lens name.
        grid_size : int
            Desired output size (will resize).
        cutout_arcsec : float, optional
            Extract a square cutout of this size. Default: 4× Einstein radius.
        allow_synthetic_fallback : bool
            If ``True``, allow a synthetic FITS surrogate when archival data are
            unavailable. Keep ``False`` for observational validation.

        Returns
        -------
        image : np.ndarray
            2D image array of shape (grid_size, grid_size).
        weight_map : np.ndarray
            2D inverse variance weight map of matching shape.
        metadata : dict
            Lens parameters and provenance.
        """
        fits_path = self.download_slacs_lens(
            name,
            allow_synthetic_fallback=allow_synthetic_fallback,
        )
        entry = self.catalog[name]

        if ASTROPY_AVAILABLE and fits_path.suffix == ".fits":
            with fits.open(fits_path) as hdul:
                sci_data, wht_data, header = None, None, {}
                for hdu in hdul:
                    if hasattr(hdu, "name"):
                        if hdu.name == "SCI":
                            sci_data = hdu.data
                            header = hdu.header
                        elif hdu.name == "WHT":
                            wht_data = hdu.data
                        elif hdu.name == "ERR":
                            # Convert error map to inverse variance weight map
                            wht_data = 1.0 / (hdu.data**2 + 1e-10)
                
                # Fallback to primary HDU if no SCI extension found
                if sci_data is None:
                    sci_data = hdul[0].data
                    header = hdul[0].header
                if wht_data is None:
                    wht_data = np.ones_like(sci_data) / (0.005**2)

                raw_image = sci_data.astype(np.float64)
                raw_weight = wht_data.astype(np.float64)
        else:
            npy_path = fits_path.with_suffix(".npy")
            wht_path = fits_path.with_name(f"{fits_path.stem}_wht.npy")
            if npy_path.exists():
                raw_image = np.load(npy_path).astype(np.float64)
                if wht_path.exists():
                    raw_weight = np.load(wht_path).astype(np.float64)
                else:
                    raw_weight = np.ones_like(raw_image) / (0.005**2)
            else:
                raise FileNotFoundError(f"No loadable data for {name}")
            header = {}

        # Center cutout around maximum (assumed lens center)
        if cutout_arcsec is None:
            cutout_arcsec = float(4.0 * entry["einstein_radius"])

        # Determine pixel scale
        pixel_scale = abs(header.get("CD1_1", 0.05 / 3600)) * 3600  # arcsec/pixel
        cutout_pixels = int(cutout_arcsec / pixel_scale)

        cy_index, cx_index = np.unravel_index(np.argmax(raw_image), raw_image.shape)
        cy = int(cy_index)
        cx = int(cx_index)
        half = int(cutout_pixels // 2)

        def _extract_and_resize(arr: np.ndarray, is_weight: bool = False) -> np.ndarray:
            y0 = int(max(0, cy - half))
            y1 = int(min(arr.shape[0], cy + half))
            x0 = int(max(0, cx - half))
            x1 = int(min(arr.shape[1], cx + half))
            cut = arr[y0:y1, x0:x1].copy()

            if cut.shape[0] != grid_size or cut.shape[1] != grid_size:
                try:
                    from scipy.ndimage import zoom
                    scale_y = grid_size / cut.shape[0]
                    scale_x = grid_size / cut.shape[1]
                    order = 1 if is_weight else 3  # Bilinear for weights, cubic for image
                    cut = zoom(cut, (scale_y, scale_x), order=order)
                except ImportError:
                    idx_y = np.linspace(0, cut.shape[0] - 1, grid_size).astype(int)
                    idx_x = np.linspace(0, cut.shape[1] - 1, grid_size).astype(int)
                    cut = cut[np.ix_(idx_y, idx_x)]
            return cut

        cutout = _extract_and_resize(raw_image)
        cutout_weight = _extract_and_resize(raw_weight, is_weight=True)

        # Normalize image to [0, 1]
        vmin, vmax = np.percentile(cutout, [1, 99])
        if vmax > vmin:
            cutout = np.clip((cutout - vmin) / (vmax - vmin), 0, 1)
            # Scale weight map inversely by normalization factor squared
            cutout_weight = cutout_weight * ((vmax - vmin) ** 2)

        output_pixel_scale = float(cutout_arcsec) / float(grid_size)
        comment_cards = header.get("COMMENT", [])
        if isinstance(comment_cards, str):
            comment_text = comment_cards
        else:
            comment_text = " ".join(str(card) for card in comment_cards)

        metadata = {
            "name": entry["name"],
            "z_lens": entry["z_lens"],
            "z_source": entry["z_source"],
            "sigma_v": entry["sigma_v"],
            "einstein_radius": entry["einstein_radius"],
            "pixel_scale_arcsec": output_pixel_scale,
            "native_pixel_scale_arcsec": float(pixel_scale),
            "cutout_arcsec": cutout_arcsec,
            "fits_path": str(fits_path),
            "is_synthetic": "Synthetic" in comment_text,
            "ref": entry["ref"],
        }

        return cutout, cutout_weight, metadata
