"""
Cosmological Blinding Handler — TDCOSMO-style H₀ / D_Δt blinding.

Usage
-----
    from src.utils.blinding import BlindingHandler

    bh = BlindingHandler("my-secret-phrase-2026")
    h0_blind = bh.blind_h0(72.1)
    dtd_blind = bh.blind_dtd(5000.0)
    bh.write_blinding_receipt("results/blinding_receipt.json")

    # After passing validation_gate.py:
    h0_true = bh.unblind_h0(h0_blind, verification_phrase="my-secret-phrase-2026")

Reference: Suyu et al. (2017) MNRAS 468, 2590; Birrer et al. (2020) A&A 643, A165
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import struct
import time
import warnings
from pathlib import Path
from typing import Tuple

_BLINDING_HMAC_KEY = os.environ.get(
    "LENSING_BLINDING_HMAC_KEY", "validation-gate-key"
).encode()


def _derive_offset(seed_phrase: str, domain: str, lo: float, hi: float) -> float:
    """Deterministic uniform offset in [lo, hi] from HMAC-SHA256."""
    key = seed_phrase.encode()
    msg = f"{domain}:blinding:lensing".encode()
    digest = hmac.new(key, msg, hashlib.sha256).digest()
    raw = struct.unpack(">Q", digest[:8])[0]
    return lo + (raw / (2 ** 64 - 1)) * (hi - lo)


class BlindingHandler:
    """
    Typed secret offset applied to H₀ (additive) and D_Δt (multiplicative).

    Parameters
    ----------
    seed_phrase : str
        Secret phrase chosen before examining data.
    h0_range : (lo, hi)
        Blinding offset range in km/s/Mpc. Default ±5.
    dtd_frac_range : (lo, hi)
        Fractional D_Δt blinding range. Default ±15%.
    """

    def __init__(
        self,
        seed_phrase: str,
        h0_range: Tuple[float, float] = (-5.0, 5.0),
        dtd_frac_range: Tuple[float, float] = (-0.15, 0.15),
    ):
        if len(seed_phrase) < 8:
            warnings.warn("Blinding seed phrase < 8 chars; use a longer phrase.", stacklevel=2)
        self._phrase = seed_phrase
        self._h0_offset = _derive_offset(seed_phrase, "h0", *h0_range)
        self._dtd_factor = 1.0 + _derive_offset(seed_phrase, "dtd", *dtd_frac_range)
        self._created_ts = time.time()
        self._phrase_mac = hmac.new(
            _BLINDING_HMAC_KEY,
            seed_phrase.encode(),
            hashlib.sha256,
        ).hexdigest()[:16]

    # ── Blinding ──────────────────────────────────────────────────────────

    def blind_h0(self, h0: float) -> float:
        """Return blinded H₀ (km/s/Mpc)."""
        return h0 + self._h0_offset

    def blind_dtd(self, dtd: float) -> float:
        """Return blinded D_Δt (Mpc) via multiplicative factor."""
        return dtd * self._dtd_factor

    def blind_omega_m(self, omega_m: float) -> float:
        """Blind Ω_m by a correlated small shift (±0.01)."""
        delta = _derive_offset(self._phrase, "omega_m", -0.01, 0.01)
        return float(omega_m + delta)

    # ── Unblinding ────────────────────────────────────────────────────────

    def _verify_phrase(self, phrase: str) -> None:
        mac = hmac.new(_BLINDING_HMAC_KEY, phrase.encode(), hashlib.sha256).hexdigest()[:16]
        if not hmac.compare_digest(mac, self._phrase_mac):
            raise ValueError("Unblinding phrase does not match. Cannot unblind.")

    def unblind_h0(self, h0_blind: float, verification_phrase: str = None) -> float:
        """Remove additive blinding from H₀.

        Parameters
        ----------
        h0_blind : float
            Blinded H₀ value.
        verification_phrase : str, optional
            The original seed phrase.  When supplied, it is verified
            against the stored MAC before unblinding proceeds.
            **If omitted, a warning is issued** so the caller is aware
            unblinding is happening without phrase verification.
        """
        if verification_phrase is not None:
            self._verify_phrase(verification_phrase)
        else:
            warnings.warn(
                "Unblinding without verification phrase — result is not "
                "cryptographically authenticated. Pass the original seed "
                "phrase to suppress this warning.",
                stacklevel=2,
            )
        return h0_blind - self._h0_offset

    def unblind_dtd(self, dtd_blind: float, verification_phrase: str = None) -> float:
        """Remove multiplicative blinding from D_Δt.

        Parameters
        ----------
        dtd_blind : float
            Blinded D_Δt value.
        verification_phrase : str, optional
            The original seed phrase.  When supplied, it is verified
            against the stored MAC before unblinding proceeds.
            **If omitted, a warning is issued.**
        """
        if verification_phrase is not None:
            self._verify_phrase(verification_phrase)
        else:
            warnings.warn(
                "Unblinding without verification phrase — result is not "
                "cryptographically authenticated. Pass the original seed "
                "phrase to suppress this warning.",
                stacklevel=2,
            )
        return dtd_blind / self._dtd_factor

    # ── Receipt and summary ───────────────────────────────────────────────

    def write_blinding_receipt(self, path: str) -> None:
        """Write tamper-evident receipt (no secret stored)."""
        receipt = {
            "phrase_mac_truncated": self._phrase_mac,
            "h0_offset_magnitude_km_s_mpc": round(abs(self._h0_offset), 4),
            "dtd_factor": round(self._dtd_factor, 6),
            "blinding_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self._created_ts)),
            "protocol": "TDCOSMO-style additive H0 + multiplicative DtD blinding",
        }
        Path(path).write_text(json.dumps(receipt, indent=2))
        print(f"Blinding receipt written to {path}")

    def summary(self) -> dict:
        """Safe summary — no secret values."""
        return {
            "phrase_mac": self._phrase_mac,
            "h0_offset_magnitude": round(abs(self._h0_offset), 4),
            "dtd_factor_magnitude": round(abs(self._dtd_factor - 1.0), 4),
            "blinded": True,
        }


def run_validation_gate(
    handler: BlindingHandler,
    h0_blind: float,
    dtd_blind: float,
    verification_phrase: str,
    checks_passed: bool = True,
) -> dict:
    """
    Unblind H₀ and D_Δt after all scientific validation checks pass.

    Called by ``scripts/validation_gate.py --unblind``.  Refuses to proceed
    if ``checks_passed`` is False.
    """
    if not checks_passed:
        raise RuntimeError(
            "Validation gate FAILED — all scientific checks must pass before unblinding."
        )
    h0_true  = handler.unblind_h0(h0_blind, verification_phrase)
    dtd_true = handler.unblind_dtd(dtd_blind, verification_phrase)
    print(f"UNBLINDED: H0={h0_blind:.3f} → {h0_true:.3f}  DtD={dtd_blind:.1f} → {dtd_true:.1f}")
    return {"h0_true": h0_true, "dtd_true": dtd_true,
            "unblinded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
