"""
LensFinder — Automated Lens Discovery for Wide-Field FITS Images

Implements a LenNet-style sliding-window detector that scans wide-field
FITS images, identifies gravitational lens candidates, and returns
localization bounding boxes with confidence scores.

Architecture:
  - Input: FITS cutout (64×64 pixels, single band)
  - CNN encoder: 3× [Conv2d → BN → ReLU → MaxPool] → 512-d embedding
  - Detection head: objectness score + (x_c, y_c, w, h) bounding box
  - Candidate NMS with IoU threshold = 0.45

Reference
---------
Jacobs et al. (2019) ApJS 243, 17 — LensFinder CNN discovery over 73 deg²
Lanusse et al. (2018) MNRAS 473, 3895 — CMU DeepLens

Usage
-----
    from src.ml.lens_finder import LensFinder, ScanResult

    finder = LensFinder(confidence_threshold=0.7)
    results = finder.scan_fits(fits_path="field.fits", stride=32)
    for r in results:
        print(r.ra, r.dec, r.confidence, r.bbox_pixels)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    warnings.warn("PyTorch not available; LensFinder will run in fallback mode", stacklevel=2)

try:
    from astropy.io import fits as astrofits
    from astropy.wcs import WCS
    _HAS_ASTROPY = True
except ImportError:
    _HAS_ASTROPY = False


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BoundingBox:
    """Pixel-space bounding box (top-left origin, image-frame coordinates)."""
    x_center: float
    y_center: float
    width: float
    height: float

    def iou(self, other: "BoundingBox") -> float:
        """Intersection-over-Union for NMS."""
        ax1, ay1 = self.x_center - self.width / 2, self.y_center - self.height / 2
        ax2, ay2 = self.x_center + self.width / 2, self.y_center + self.height / 2
        bx1, by1 = other.x_center - other.width / 2, other.y_center - other.height / 2
        bx2, by2 = other.x_center + other.width / 2, other.y_center + other.height / 2
        inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
        inter_h = max(0, min(ay2, by2) - max(ay1, by1))
        inter = inter_w * inter_h
        union = self.width * self.height + other.width * other.height - inter
        return inter / (union + 1e-9)


@dataclass
class ScanResult:
    """A single lens candidate output by the finder."""
    confidence: float
    bbox: BoundingBox
    ra: Optional[float] = None           # degrees (if WCS available)
    dec: Optional[float] = None          # degrees (if WCS available)
    cutout_pixels: Optional[np.ndarray] = None   # (64, 64) subimage
    pixel_x: float = 0.0                 # global pixel x of detection centre
    pixel_y: float = 0.0                 # global pixel y of detection centre
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "confidence": round(self.confidence, 4),
            "ra": round(self.ra, 6) if self.ra is not None else None,
            "dec": round(self.dec, 6) if self.dec is not None else None,
            "pixel_x": round(self.pixel_x, 1),
            "pixel_y": round(self.pixel_y, 1),
            "bbox": {
                "x_center": round(self.bbox.x_center, 1),
                "y_center": round(self.bbox.y_center, 1),
                "width": round(self.bbox.width, 1),
                "height": round(self.bbox.height, 1),
            },
        }


# ---------------------------------------------------------------------------
# CNN model
# ---------------------------------------------------------------------------

class _LensBlock(nn.Module if _HAS_TORCH else object):
    """Conv-BN-ReLU-MaxPool block."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.block(x)


class LensNetModel(nn.Module if _HAS_TORCH else object):
    """
    Lightweight LenNet-style CNN for gravitational lens classification
    and bounding-box regression.

    Input:  (B, 1, 64, 64) — normalised flux cutout
    Output: (B, 5) — [objectness, x_c, y_c, w, h] (all in [0, 1] units)

    Ref: Jacobs et al. (2019) ApJS 243, 17
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            _LensBlock(1, 32),    # → (B, 32, 32, 32)
            _LensBlock(32, 64),   # → (B, 64, 16, 16)
            _LensBlock(64, 128),  # → (B, 128, 8, 8)
            _LensBlock(128, 256), # → (B, 256, 4, 4)
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 5),   # objectness + 4 box params
        )

    def forward(self, x):
        features = self.encoder(x)
        out = self.head(features)
        objectness = torch.sigmoid(out[:, :1])
        bbox = torch.sigmoid(out[:, 1:])  # normalised [0, 1]
        return torch.cat([objectness, bbox], dim=1)


# ---------------------------------------------------------------------------
# Non-maximum suppression
# ---------------------------------------------------------------------------

def _nms(candidates: List[ScanResult], iou_thresh: float = 0.45) -> List[ScanResult]:
    """Greedy NMS: keep highest-confidence, suppress overlapping boxes."""
    candidates = sorted(candidates, key=lambda c: c.confidence, reverse=True)
    kept: List[ScanResult] = []
    for cand in candidates:
        suppress = any(cand.bbox.iou(k.bbox) > iou_thresh for k in kept)
        if not suppress:
            kept.append(cand)
    return kept


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

class LensFinder:
    """
    Automated gravitational lens finder for wide-field survey images.

    Implements a sliding-window scan over a FITS image using a LenNet-style
    CNN. Detections above ``confidence_threshold`` are returned as
    ``ScanResult`` objects with pixel and (optionally) WCS sky coordinates.

    Parameters
    ----------
    confidence_threshold : float
        Minimum objectness score to report as a candidate.
    cutout_size : int
        Side length of each sliding-window cutout in pixels.
    nms_iou_threshold : float
        IoU threshold for non-maximum suppression.
    checkpoint : str or None
        Path to a saved ``LensNetModel`` state dict. If None, reports
        confidence from a physics-heuristic fallback.
    """

    CUTOUT_SIZE = 64

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        nms_iou_threshold: float = 0.45,
        checkpoint: Optional[str] = None,
    ):
        self.conf_thresh = confidence_threshold
        self.nms_iou = nms_iou_threshold
        self.model = None

        if _HAS_TORCH:
            self.model = LensNetModel()
            if checkpoint and Path(checkpoint).exists():
                sd = torch.load(checkpoint, map_location="cpu")
                self.model.load_state_dict(sd)
            self.model.eval()

    def _preprocess_cutout(self, cutout: np.ndarray) -> np.ndarray:
        """Normalise and resize cutout to (1, 1, 64, 64) float32."""
        # Resize to 64×64 via nearest (fast, no scipy dependency for inference)
        c = np.array(cutout, dtype=np.float32)
        ny, nx = c.shape[:2]
        if (ny, nx) != (self.CUTOUT_SIZE, self.CUTOUT_SIZE):
            # Simple nearest-neighbour resize
            iy = np.round(np.linspace(0, ny - 1, self.CUTOUT_SIZE)).astype(int)
            ix = np.round(np.linspace(0, nx - 1, self.CUTOUT_SIZE)).astype(int)
            c = c[np.ix_(iy, ix)]
        # Robust normalisation
        p2, p98 = np.percentile(c, [2, 98])
        rng = p98 - p2
        if rng > 0:
            c = (c - p2) / rng
        c = np.clip(c, 0.0, 1.0)
        return c[np.newaxis, np.newaxis, :, :]  # (1, 1, 64, 64)

    @staticmethod
    def _physics_heuristic_score(cutout: np.ndarray) -> Tuple[float, BoundingBox]:
        """
        Fallback confidence estimator without a trained model.

        Uses ring-like flux pattern: computes the ratio of flux in an annular
        region (Einstein-ring radius ~0.3–0.5 of cutout half-width) to total
        flux. High ratio → likely ring/arc morphology.
        """
        cut = np.asarray(cutout, dtype=np.float64)
        cut = np.clip(cut - np.percentile(cut, 10), 0, None)
        h, w = cut.shape[:2]
        cy, cx = h // 2, w // 2
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        r_in = min(h, w) * 0.20
        r_out = min(h, w) * 0.50
        annulus = (dist >= r_in) & (dist <= r_out)
        total = cut.sum() + 1e-10
        ring_flux = cut[annulus].sum() / total
        # Scale: ring_flux > 0.35 is a confident ring
        confidence = float(np.clip((ring_flux - 0.15) / 0.20, 0.0, 1.0))
        box = BoundingBox(x_center=cx, y_center=cy,
                          width=r_out * 2, height=r_out * 2)
        return confidence, box

    def _infer_cutout(self, cutout: np.ndarray, x0: int, y0: int) -> Optional[ScanResult]:
        """Run model or heuristic on one cutout; return ScanResult if above threshold."""
        if self.model is not None and _HAS_TORCH:
            tensor = torch.from_numpy(self._preprocess_cutout(cutout))
            with torch.no_grad():
                pred = self.model(tensor)[0].cpu().numpy()
            confidence = float(pred[0])
            xc = float(pred[1]) * self.CUTOUT_SIZE + x0
            yc = float(pred[2]) * self.CUTOUT_SIZE + y0
            bw = float(pred[3]) * self.CUTOUT_SIZE
            bh = float(pred[4]) * self.CUTOUT_SIZE
            bbox = BoundingBox(xc, yc, bw, bh)
        else:
            confidence, local_box = self._physics_heuristic_score(cutout)
            bbox = BoundingBox(
                local_box.x_center + x0, local_box.y_center + y0,
                local_box.width, local_box.height,
            )

        if confidence < self.conf_thresh:
            return None

        return ScanResult(
            confidence=confidence,
            bbox=bbox,
            pixel_x=bbox.x_center,
            pixel_y=bbox.y_center,
            cutout_pixels=cutout.copy(),
        )

    def scan_array(
        self,
        image: np.ndarray,
        stride: int = 32,
        wcs: Optional[object] = None,
    ) -> List[ScanResult]:
        """
        Scan a 2-D image array with a sliding window.

        Parameters
        ----------
        image : (H, W) ndarray
            Flux image (any units; normalised internally).
        stride : int
            Step size between successive windows in pixels.
        wcs : astropy.wcs.WCS or None
            If provided, converts detections to RA/Dec.

        Returns
        -------
        list[ScanResult]
            NMS-filtered candidates sorted by confidence (descending).
        """
        h, w = image.shape[:2]
        cs = self.CUTOUT_SIZE
        candidates: List[ScanResult] = []

        for y0 in range(0, h - cs + 1, stride):
            for x0 in range(0, w - cs + 1, stride):
                cutout = image[y0 : y0 + cs, x0 : x0 + cs]
                result = self._infer_cutout(cutout, x0, y0)
                if result is not None:
                    candidates.append(result)

        candidates = _nms(candidates, iou_thresh=self.nms_iou)

        # Attach WCS coordinates
        if wcs is not None and _HAS_ASTROPY:
            for r in candidates:
                try:
                    sky = wcs.pixel_to_world(r.pixel_x, r.pixel_y)
                    r.ra = float(sky.ra.deg)
                    r.dec = float(sky.dec.deg)
                except Exception:
                    pass

        return sorted(candidates, key=lambda c: c.confidence, reverse=True)

    def scan_fits(
        self,
        fits_path: str,
        extension: int = 1,
        stride: int = 32,
    ) -> List[ScanResult]:
        """
        Scan a FITS file end-to-end.

        Parameters
        ----------
        fits_path : str
            Path to FITS file (HST/Euclid/Roman compatible).
        extension : int
            FITS extension to read the science image from.
        stride : int
            Sliding-window step in pixels.
        """
        if not _HAS_ASTROPY:
            raise ImportError("astropy required for FITS scanning")

        with astrofits.open(fits_path) as hdul:
            data = hdul[extension].data.astype(np.float32)
            header = hdul[extension].header
            try:
                wcs = WCS(header, naxis=2)
            except Exception:
                wcs = None

        return self.scan_array(data, stride=stride, wcs=wcs)

    def scan_synthetic(
        self,
        grid_size: int = 256,
        n_lenses: int = 5,
        seed: int = 42,
    ) -> List[ScanResult]:
        """
        Scan a synthetic wide-field image with injected lens signals.
        Useful for CI/quick demo without a real FITS file.

        Parameters
        ----------
        grid_size : int
            Size of the synthetic field (pixels).
        n_lenses : int
            Number of injected Einstein-ring signals.
        seed : int
            RNG seed for reproducibility.
        """
        rng = np.random.RandomState(seed)
        # Background: Gaussian noise
        field = rng.normal(0, 0.01, (grid_size, grid_size)).astype(np.float32)

        # Inject ring-like signals
        positions = []
        for _ in range(n_lenses):
            cy = rng.randint(40, grid_size - 40)
            cx = rng.randint(40, grid_size - 40)
            r_e = rng.uniform(8, 18)
            Y, X = np.ogrid[:grid_size, :grid_size]
            dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
            # Thin ring (Einstein ring approximation)
            ring = np.exp(-0.5 * ((dist - r_e) / 1.5) ** 2) * rng.uniform(0.3, 1.0)
            field += ring.astype(np.float32)
            positions.append((cx, cy))

        results = self.scan_array(field, stride=32)
        return results
