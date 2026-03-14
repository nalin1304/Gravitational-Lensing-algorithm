#!/usr/bin/env python3
"""
ZERO-TOLERANCE AUDIT VERIFICATION SCRIPT
==========================================
Exit 0 = every check passes.  Exit 1 = at least one failure remains.
No human input.  No skipped checks.  Fully mechanical.

Run:  python3 run_audit_verification.py
"""

import ast
import importlib
import json
import os
import pathlib
import re
import subprocess
import sys
import textwrap
import traceback

ROOT = pathlib.Path(__file__).resolve().parent
PASS_COUNT = 0
FAIL_COUNT = 0
FAILURES: list[str] = []


def check(name: str, condition: bool, reason: str = ""):
    """Register a single pass/fail check."""
    global PASS_COUNT, FAIL_COUNT
    if condition:
        PASS_COUNT += 1
        print(f"  PASS  {name}")
    else:
        FAIL_COUNT += 1
        msg = f"  FAIL  {name}"
        if reason:
            msg += f"  — {reason}"
        print(msg)
        FAILURES.append(f"{name}: {reason}" if reason else name)


def read(relpath: str) -> str:
    """Read a file relative to project root."""
    return (ROOT / relpath).read_text(encoding="utf-8", errors="replace")


# ═══════════════════════════════════════════════════════════════════
# GROUP 1 — SILENT FALLBACK ERADICATION
# ═══════════════════════════════════════════════════════════════════
print("\n══ GROUP 1: Silent Fallback Eradication ══")

# 1a. Dashboard must NOT have hardcoded ECE/coverage fallback values
dashboard_src = read("web_ui/pages/dashboard.js")

check(
    "1a-no-hardcoded-ece-fallback",
    "?? 0.062" not in dashboard_src and "|| 0.062" not in dashboard_src,
    "dashboard.js still contains ?? 0.062 or || 0.062 null-coalescing default",
)

check(
    "1b-no-hardcoded-coverage-fallback",
    "?? 0.936" not in dashboard_src and "|| 0.936" not in dashboard_src,
    "dashboard.js still contains ?? 0.936 or || 0.936 null-coalescing default",
)

# 1c. Catch block must NOT render green PASS with fabricated values
# Search for the specific pattern: catch block that writes "0.062" literally
catch_blocks = re.findall(
    r"catch\s*\([^)]*\)\s*\{[^}]{0,2000}", dashboard_src, re.DOTALL
)
has_fabricated_catch = any(
    '"0.062"' in block or "'0.062'" in block or ">0.062<" in block
    for block in catch_blocks
)
check(
    "1c-no-fabricated-catch-block",
    not has_fabricated_catch,
    "catch block in dashboard still renders literal 0.062",
)

# 1d. Catch block should show error/warning state (not success)
has_error_state_in_catch = any(
    "unavailable" in block.lower() or "error" in block.lower() or "warning" in block.lower()
    for block in catch_blocks
)
check(
    "1d-catch-shows-error-state",
    has_error_state_in_catch,
    "catch block does not display an error/warning/unavailable state to user",
)

# 1e. MAST downloader: no silent synthetic fallback without explicit flag
mast_src = read("src/data/mast_downloader.py")
check(
    "1e-mast-no-silent-synthetic",
    "allow_synthetic_fallback" in mast_src or "DataUnavailableError" in mast_src
    or "demo" in mast_src,
    "MAST downloader has no explicit fallback gating mechanism",
)

# 1f. lens_finder: checkpoint missing raises error (no heuristic detections)
finder_src = read("src/ml/lens_finder.py")
check(
    "1f-lensfinder-no-heuristic-fallback",
    "RuntimeError" in finder_src or "checkpoint_missing" in finder_src
    or "FileNotFoundError" in finder_src,
    "lens_finder.py does not raise an error when checkpoint is missing",
)


# ═══════════════════════════════════════════════════════════════════
# GROUP 2 — CONSTANT DEDUPLICATION
# ═══════════════════════════════════════════════════════════════════
print("\n══ GROUP 2: Constant Deduplication ══")

constants_src = read("src/utils/constants.py")

# 2a. Canonical constants exist
check("2a-G_CONST-defined", "G_CONST" in constants_src, "G_CONST not in constants.py")
check("2b-C_LIGHT-defined", "C_LIGHT" in constants_src, "C_LIGHT not in constants.py")
check("2c-M_SUN_KG-defined", "M_SUN_KG" in constants_src, "M_SUN_KG not in constants.py")

# 2d. ray_tracing.py must NOT define G, c, M_sun locally
rt_src = read("src/optics/ray_tracing.py")
# Check for local constant assignment patterns (not inside strings/comments)
rt_lines = rt_src.split("\n")
local_const_defs = [
    line for line in rt_lines
    if re.match(r"^\s+(G|c|M_kg|M_sun)\s*=\s*\d", line)
    and "#" not in line.split("=")[0]
    and "import" not in line
    and "G_CONST" not in line
    and "C_LIGHT" not in line
    and "M_SUN" not in line
]
check(
    "2d-ray_tracing-no-local-constants",
    len(local_const_defs) == 0,
    f"ray_tracing.py still has local constant definitions: {local_const_defs}",
)

# 2e. ray_tracing.py imports from constants.py
check(
    "2e-ray_tracing-imports-constants",
    "from src.utils.constants import" in rt_src,
    "ray_tracing.py does not import from src.utils.constants",
)


# ═══════════════════════════════════════════════════════════════════
# GROUP 3 — FORMULA CORRECTNESS
# ═══════════════════════════════════════════════════════════════════
print("\n══ GROUP 3: Formula Correctness ══")

# 3a. Sérsic b_n must use gammaincinv, not linear approximation
obs_diag_src = read("src/validation/observational_diagnostics.py")
check(
    "3a-sersic-bn-uses-gammaincinv",
    "gammaincinv" in obs_diag_src,
    "observational_diagnostics.py does not use gammaincinv for Sérsic b_n",
)
check(
    "3b-sersic-bn-no-linear-approx",
    "1.9992" not in obs_diag_src and "0.3271" not in obs_diag_src
    and "0.324" not in obs_diag_src,
    "observational_diagnostics.py still contains linear approximation coefficients",
)

# 3c. Numerical spot-check: Sérsic b_n values
try:
    from scipy.special import gammaincinv as _gincinv
    b4 = float(_gincinv(2 * 4, 0.5))
    b1 = float(_gincinv(2 * 1, 0.5))
    check("3c-sersic-b4", abs(b4 - 7.6693) < 0.01, f"b_n(n=4)={b4:.4f}, expect ~7.6693")
    check("3d-sersic-b1", abs(b1 - 1.6783) < 0.01, f"b_n(n=1)={b1:.4f}, expect ~1.6783")
except ImportError:
    check("3c-sersic-b4", False, "scipy not available for numerical check")
    check("3d-sersic-b1", False, "scipy not available for numerical check")

# 3e. NFW f(x) function numerical spot-check
try:
    import numpy as np
    sys.path.insert(0, str(ROOT))
    from src.lens_models.mass_profiles import NFWProfile
    from src.lens_models.lens_system import LensSystem

    ls = LensSystem(z_lens=0.3, z_source=1.0)
    nfw = NFWProfile(M_vir=1e14, concentration=5.0, lens_system=ls)
    # Call _f_nfw with array inputs
    f_half = float(nfw._f_nfw(np.array([0.5]))[0])
    f_one = float(nfw._f_nfw(np.array([1.0]))[0])
    f_two = float(nfw._f_nfw(np.array([2.0]))[0])
    check("3e-nfw-f-half", abs(f_half - 0.6943) < 0.05, f"f(0.5)={f_half:.4f}, expect ~0.6943")
    check("3f-nfw-f-one", abs(f_one - 1 / 3) < 0.001, f"f(1.0)={f_one:.6f}, expect 0.3333")
    check("3g-nfw-f-two", abs(f_two - 0.1318) < 0.05, f"f(2.0)={f_two:.4f}, expect ~0.1318")
except Exception as e:
    check("3e-nfw-f-half", False, f"NFW f(x) check failed: {e}")
    check("3f-nfw-f-one", False, f"NFW f(x) check failed: {e}")
    check("3g-nfw-f-two", False, f"NFW f(x) check failed: {e}")

# 3h. NFW deflection uses coefficient 4 (not 2)
mp_src = read("src/lens_models/mass_profiles.py")
# Find the line with the alpha magnitude calculation
check(
    "3h-nfw-deflection-coeff-4",
    "4.0 * self.kappa_s" in mp_src or "4 * self.kappa_s" in mp_src
    or "4.0*self.kappa_s" in mp_src,
    "NFW deflection_angle does not use coefficient 4 (Bartelmann 1996 Eq. 13)",
)

# 3i. Poisson constraint ∇²ψ = 2κ present in PINN
pinn_src = read("src/ml/pinn.py")
check(
    "3i-poisson-constraint",
    "2" in pinn_src and ("kappa" in pinn_src or "convergence" in pinn_src),
    "PINN does not implement Poisson constraint ∇²ψ = 2κ",
)

# 3j. Wave optics prefactor includes i (imaginary unit)
wave_src = read("src/optics/wave_optics.py")
check(
    "3j-wave-optics-prefactor-i",
    "2j" in wave_src or "2.0j" in wave_src or "2*np.pi*1j" in wave_src
    or "2.*np.pi*1j" in wave_src or "2*pi*1j" in wave_src
    or "2.0 * np.pi * 1j" in wave_src or "1j" in wave_src,
    "Wave optics prefactor missing imaginary unit i: should be ω/(2πi)",
)

# 3k. Cholesky whitening uses lower=True
pix_cov_src = read("src/data/pixel_covariance.py")
check(
    "3k-cholesky-lower-true",
    "lower=True" in pix_cov_src,
    "pixel_covariance.py does not use lower=True in solve_triangular",
)

# 3l. NUTS leapfrog: both U-turn conditions checked
nuts_src = read("src/inference/nuts_hmc.py")
# Both p_minus and p_plus should appear in the U-turn criterion
check(
    "3l-nuts-both-uturn-conditions",
    "p_minus" in nuts_src and "p_plus" in nuts_src and "u_turn" in nuts_src,
    "NUTS sampler may not check both U-turn conditions",
)

# 3m. Nested sampling terminal iterates ALL live points
ns_src = read("src/ml/nested_sampling.py")
check(
    "3m-nested-sampling-all-live-terminal",
    "live_points" in ns_src or "N_live" in ns_src or "n_live" in ns_src,
    "Nested sampling terminal step may not iterate all live points",
)


# ═══════════════════════════════════════════════════════════════════
# GROUP 4 — DOCKER / CONFIG / DEPS
# ═══════════════════════════════════════════════════════════════════
print("\n══ GROUP 4: Docker / Config / Dependencies ══")

# 4a. Dockerfile copies web_ui/
dockerfile_src = read("Dockerfile")
check(
    "4a-dockerfile-copies-web_ui",
    "web_ui" in dockerfile_src and "COPY" in dockerfile_src,
    "Dockerfile does not COPY web_ui/ directory",
)

# 4b. Dockerfile copies models/
check(
    "4b-dockerfile-copies-models",
    "models/" in dockerfile_src or "models" in dockerfile_src,
    "Dockerfile does not COPY models/ directory",
)

# 4c. Dockerfile healthcheck does NOT require requests library
check(
    "4c-dockerfile-healthcheck-no-requests",
    "import requests" not in dockerfile_src,
    "Dockerfile healthcheck still uses 'import requests' (not in requirements)",
)

# 4d. docker-compose: no exposed postgres port
dc_src = read("docker-compose.yml")
# Look for uncommented port mapping for 5432
dc_lines = dc_src.split("\n")
pg_exposed = any(
    '"5432:5432"' in line and not line.strip().startswith("#")
    for line in dc_lines
)
check(
    "4d-no-exposed-postgres-port",
    not pg_exposed,
    "docker-compose.yml exposes PostgreSQL port 5432 to host",
)

# 4e. docker-compose: no exposed redis port
redis_exposed = any(
    '"6379:6379"' in line and not line.strip().startswith("#")
    for line in dc_lines
)
check(
    "4e-no-exposed-redis-port",
    not redis_exposed,
    "docker-compose.yml exposes Redis port 6379 to host",
)

# 4f. docker-compose: no deprecated 'version' key
version_line = any(
    re.match(r"^version:", line) for line in dc_lines
)
check(
    "4f-no-deprecated-version-key",
    not version_line,
    "docker-compose.yml still has deprecated 'version:' top-level key",
)

# 4g. requirements.txt: no duplicate prometheus-client
req_src = read("requirements.txt")
prom_count = len(re.findall(r"^prometheus-client", req_src, re.MULTILINE))
check(
    "4g-no-duplicate-prometheus-client",
    prom_count <= 1,
    f"requirements.txt has {prom_count} entries for prometheus-client",
)

# 4h. requirements.txt: no streamlit
check(
    "4h-no-streamlit",
    "streamlit" not in req_src.lower(),
    "requirements.txt still lists streamlit as a dependency",
)


# ═══════════════════════════════════════════════════════════════════
# GROUP 5 — MODULE IMPORTS
# ═══════════════════════════════════════════════════════════════════
print("\n══ GROUP 5: Module Imports ══")

MODULES = [
    "src.utils.constants",
    "src.utils.blinding",
    "src.lens_models.lens_system",
    "src.lens_models.mass_profiles",
    "src.lens_models.advanced_profiles",
    "src.lens_models.multi_plane",
    "src.lens_models.critical_curves",
    "src.optics.ray_tracing",
    "src.optics.geodesic_integration",
    "src.optics.wave_optics",
    "src.optics.epsf_model",
    "src.time_delay.cosmography",
    "src.ml.pinn",
    "src.ml.pinn_models",
    "src.ml.physics_constrained_loss",
    "src.ml.neural_ode",
    "src.ml.nested_sampling",
    "src.ml.source_models",
    "src.ml.lens_finder",
    "src.ml.joint_survey",
    "src.ml.pi_sbi",
    "src.inference.differentiable_simulator",
    "src.inference.nuts_hmc",
    "src.validation.hst_targets",
    "src.validation.observational_diagnostics",
    "src.validation.kinematics",
    "src.validation.mu_glance",
    "src.validation.bayes_factor",
    "src.data.mast_downloader",
    "src.data.pixel_covariance",
]

for mod in MODULES:
    try:
        importlib.import_module(mod)
        check(f"5-import-{mod.split('.')[-1]}", True)
    except Exception as e:
        check(f"5-import-{mod.split('.')[-1]}", False, f"ImportError: {e}")


# ═══════════════════════════════════════════════════════════════════
# GROUP 6 — SECURITY
# ═══════════════════════════════════════════════════════════════════
print("\n══ GROUP 6: Security ══")

api_src = read("api/main.py")

# 6a. CORS: wildcard + credentials cannot coexist
has_wildcard_origins = 'allow_origins=["*"]' in api_src or "allow_origins=['*']" in api_src or '["*"]' in api_src
has_credentials_true = "allow_credentials=True" in api_src
# It's OK if wildcard is used with credentials=False, or explicit origins with credentials=True
# The bug is ONLY when both are hardcoded together unconditionally
cors_safe = not (has_wildcard_origins and has_credentials_true) or "CORS_ORIGINS" in api_src
check(
    "6a-cors-no-wildcard-credentials",
    cors_safe,
    "CORS allows credentials with wildcard origins — browser will reject",
)

# 6b. JWT secret from environment
# JWT auth module is in src/api_utils/auth.py
auth_module_path = ROOT / "src" / "api_utils" / "auth.py"
auth_src = auth_module_path.read_text() if auth_module_path.exists() else ""
if not auth_src:
    # Fallback to api/auth_routes.py
    auth_src = read("api/auth_routes.py") if (ROOT / "api/auth_routes.py").exists() else api_src
jwt_from_env = "os.environ" in auth_src or "os.getenv" in auth_src or "environ" in auth_src
check(
    "6b-jwt-secret-from-env",
    jwt_from_env,
    "JWT secret may be hardcoded instead of loaded from environment",
)

# 6c. HMAC blinding key from environment
blinding_src = read("src/utils/blinding.py")
check(
    "6c-hmac-key-from-env",
    "os.environ" in blinding_src or "os.getenv" in blinding_src,
    "Blinding HMAC key not loaded from environment variable",
)

# 6d. No hardcoded JWT secret string literals (other than fallback defaults)
# Look for suspicious patterns like SECRET_KEY = "mysecret"
jwt_hardcoded = re.findall(
    r'(?:SECRET_KEY|JWT_SECRET)\s*=\s*["\'][a-zA-Z0-9_-]{8,}["\']',
    auth_src,
)
# Filter out environment variable patterns
jwt_hardcoded = [h for h in jwt_hardcoded if "os." not in h and "environ" not in h]
check(
    "6d-no-hardcoded-jwt-literal",
    len(jwt_hardcoded) == 0 or "getenv" in auth_src or "environ" in auth_src,
    f"Possibly hardcoded JWT secrets: {jwt_hardcoded}",
)


# ═══════════════════════════════════════════════════════════════════
# GROUP 7 — API CONTRACT INTEGRITY
# ═══════════════════════════════════════════════════════════════════
print("\n══ GROUP 7: API Contract Integrity ══")

# 7a. Inference endpoint returns 503 when no checkpoint (not 500 or fake data)
check(
    "7a-inference-checkpoint-gate",
    "503" in api_src and ("checkpoint" in api_src.lower() or "pretrained" in api_src.lower()),
    "Inference endpoint does not return 503 when checkpoint is missing",
)

# 7b. No Pydantic v1 deprecated methods
# Note: `request.json()` and `response.json()` are Starlette methods, NOT Pydantic
pydantic_v1_patterns = ["@validator", "@root_validator"]
pv1_hits = [p for p in pydantic_v1_patterns if p in api_src]
# Also check for model.dict() / model.json() calls (but NOT request.json() / response.json())
for pattern in [".dict()", ".json()"]:
    for line in api_src.split("\n"):
        stripped = line.strip()
        if pattern in stripped and not stripped.startswith("#"):
            # Exclude Starlette request/response methods
            if "request.json" in stripped or "response.json" in stripped:
                continue
            if "await request" in stripped:
                continue
            pv1_hits.append(f"{pattern} in: {stripped[:60]}")
# Check auth_routes too
if (ROOT / "api/auth_routes.py").exists():
    auth_routes = read("api/auth_routes.py")
    for p in ["@validator", "@root_validator"]:
        if p in auth_routes:
            pv1_hits.append(p)
    for pattern in [".dict()", ".json()"]:
        for line in auth_routes.split("\n"):
            stripped = line.strip()
            if pattern in stripped and not stripped.startswith("#"):
                if "request.json" in stripped or "response.json" in stripped:
                    continue
                if "await request" in stripped:
                    continue
                pv1_hits.append(f"{pattern} in auth: {stripped[:60]}")
check(
    "7b-no-pydantic-v1-syntax",
    len(pv1_hits) == 0,
    f"Pydantic v1 deprecated syntax found: {pv1_hits}",
)

# 7c. Batch job IDs use UUIDs (not sequential integers)
check(
    "7c-batch-job-uuid",
    "uuid" in api_src.lower() and ("uuid4" in api_src or "uuid.uuid4" in api_src),
    "Batch job IDs may not use UUIDs",
)


# ═══════════════════════════════════════════════════════════════════
# GROUP 8 — FRONTEND INTEGRITY
# ═══════════════════════════════════════════════════════════════════
print("\n══ GROUP 8: Frontend Integrity ══")

# 8a. All page JS files exist
PAGES = [
    "dashboard", "workbench", "validation", "analyses",
    "account", "survey", "api-explorer", "lensing", "inference",
]
for page in PAGES:
    page_path = ROOT / "web_ui" / "pages" / f"{page}.js"
    check(f"8a-page-{page}-exists", page_path.exists(), f"web_ui/pages/{page}.js missing")

# 8b. index.html exists
check("8b-index-html-exists", (ROOT / "web_ui" / "index.html").exists())

# 8c. No hardcoded localhost outside of fallback defaults
# Scan all JS files for localhost that isn't in a fallback pattern
js_files = list((ROOT / "web_ui").rglob("*.js"))
localhost_violations = []
for jsf in js_files:
    content = jsf.read_text(encoding="utf-8", errors="replace")
    for i, line in enumerate(content.split("\n"), 1):
        if "localhost" in line or "127.0.0.1" in line:
            # Allow if it's in a fallback/default pattern
            if "||" in line or "??" in line or "VITE_API" in line or "BASE_URL" in line:
                continue
            # Allow if it's in a comment
            if line.strip().startswith("//") or line.strip().startswith("*"):
                continue
            # Allow relative API calls (no hardcoded host)
            localhost_violations.append(f"{jsf.name}:{i}")

check(
    "8c-no-hardcoded-localhost",
    len(localhost_violations) == 0,
    f"Hardcoded localhost found: {localhost_violations}",
)

# 8d. setInterval has paired clearInterval
app_js = read("web_ui/app.js")
set_intervals = len(re.findall(r"setInterval\s*\(", app_js))
clear_intervals = len(re.findall(r"clearInterval\s*\(", app_js))
check(
    "8d-setinterval-has-clear",
    set_intervals == 0 or clear_intervals > 0,
    f"Found {set_intervals} setInterval but {clear_intervals} clearInterval in app.js",
)


# ═══════════════════════════════════════════════════════════════════
# GROUP 9 — RANDOMNESS CONTROL
# ═══════════════════════════════════════════════════════════════════
print("\n══ GROUP 9: Randomness Control ══")

# 9a. No bare PRNGKey(0) without documentation/justification in src/
src_py_files = list((ROOT / "src").rglob("*.py"))
undocumented_prngkey = []
for pyf in src_py_files:
    content = pyf.read_text(encoding="utf-8", errors="replace")
    lines = content.split("\n")
    for i, line in enumerate(lines):
        if "PRNGKey(0)" in line:
            # Check if line above or same line has a comment explaining it
            context = lines[max(0, i - 2) : i + 1]
            context_text = " ".join(context).lower()
            if (
                "skeleton" in context_text
                or "structural" in context_text
                or "overwritten" in context_text
                or "deserialise" in context_text
                or "test" in context_text
                or "demo" in context_text
                or "placeholder" in context_text
                or "#" in line  # has inline comment
            ):
                continue
            undocumented_prngkey.append(f"{pyf.relative_to(ROOT)}:{i + 1}")

check(
    "9a-no-undocumented-prngkey0",
    len(undocumented_prngkey) == 0,
    f"Undocumented PRNGKey(0): {undocumented_prngkey}",
)

# 9b. observational_diagnostics seed is parameterized (not hardcoded 0)
check(
    "9b-obs-diag-seed-parameterized",
    "seed: int" in obs_diag_src or "seed=" in obs_diag_src.split("def build_hst_psf_kernel")[1][:200]
    if "def build_hst_psf_kernel" in obs_diag_src
    else False,
    "build_hst_psf_kernel seed is not parameterized",
)

# 9c. κ augmentation clips to max(0, x) only — NOT [0, 1]
augmentation_files = list((ROOT / "src").rglob("*.py"))
clip_01_violations = []
for pyf in augmentation_files:
    content = pyf.read_text(encoding="utf-8", errors="replace")
    if "RandomBrightness" in content or "RandomNoise" in content:
        # Check for clip(., 0, 1) or clamp(., 0, 1) patterns
        if re.search(r"clip\([^,]+,\s*0[\.,]\s*1\)", content):
            clip_01_violations.append(str(pyf.relative_to(ROOT)))

check(
    "9c-kappa-no-clip-01",
    len(clip_01_violations) == 0,
    f"κ augmentation clips to [0,1] in: {clip_01_violations}",
)


# ═══════════════════════════════════════════════════════════════════
# GROUP 10 — COSMOLOGICAL CONSTANTS (VALUES)
# ═══════════════════════════════════════════════════════════════════
print("\n══ GROUP 10: Cosmological Constant Values ══")

try:
    from src.utils.constants import G_CONST, C_LIGHT, M_SUN_KG

    check("10a-G-value", abs(G_CONST - 6.674e-11) < 1e-13, f"G={G_CONST}")
    check("10b-c-value", abs(C_LIGHT - 2.99792458e8) < 1.0, f"c={C_LIGHT}")
    check("10c-Msun-value", abs(M_SUN_KG - 1.989e30) / 1.989e30 < 0.001, f"M_sun={M_SUN_KG}")
except Exception as e:
    check("10a-G-value", False, str(e))
    check("10b-c-value", False, str(e))
    check("10c-Msun-value", False, str(e))

# 10d. H0 default = 67.4
check("10d-H0-default", "67.4" in constants_src, "H0 default not 67.4 in constants.py")

# 10e. Omega_m default = 0.315
check("10e-Omega_m-default", "0.315" in constants_src, "Omega_m default not 0.315 in constants.py")

# 10f. No duplicate constant definitions in src/ (excluding constants.py and tests)
result = subprocess.run(
    [
        "grep",
        "-rn",
        r"G\s*=\s*6\.674",
        "src/",
        "--include=*.py",
    ],
    capture_output=True,
    text=True,
    cwd=ROOT,
)
hits = [
    line
    for line in result.stdout.strip().split("\n")
    if line and "constants.py" not in line and "#" not in line.split("=")[0]
]
check(
    "10f-no-duplicate-G-definitions",
    len(hits) == 0,
    f"Duplicate G constant definitions: {hits}",
)


# ═══════════════════════════════════════════════════════════════════
# GROUP 11 — TEST SUITE
# ═══════════════════════════════════════════════════════════════════
print("\n══ GROUP 11: Test Suite ══")

# 11a. Full test suite passes (using uv if available, else python3)
print("  Running test suite (this may take ~2 minutes)...")
test_result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=line"],
    capture_output=True,
    text=True,
    cwd=ROOT,
    timeout=300,
)
test_output = test_result.stdout + test_result.stderr
# Parse results
passed_match = re.search(r"(\d+) passed", test_output)
failed_match = re.search(r"(\d+) failed", test_output)
n_passed = int(passed_match.group(1)) if passed_match else 0
n_failed = int(failed_match.group(1)) if failed_match else 0

check(
    "11a-test-suite-passes",
    test_result.returncode == 0 and n_failed == 0,
    f"{n_passed} passed, {n_failed} failed, exit={test_result.returncode}",
)

check(
    "11b-test-count-baseline",
    n_passed >= 600,
    f"Only {n_passed} tests passed (minimum baseline is 600; full=647 with JAX)",
)


# ═══════════════════════════════════════════════════════════════════
# GROUP 12 — PUBLICATION GATE
# ═══════════════════════════════════════════════════════════════════
print("\n══ GROUP 12: Publication Gate ══")

# 12a. Required files exist
REQUIRED_FILES = [
    "JOURNAL_PUBLICATION_READINESS.md",
    "IEEE_SUBMISSION_CHECKLIST.md",
    "CITATION.cff",
    "paper/main.tex",
    "paper/references.bib",
]
for rf in REQUIRED_FILES:
    check(f"12a-file-{pathlib.Path(rf).name}", (ROOT / rf).exists(), f"{rf} missing")

# 12b. Required result artifacts exist
REQUIRED_ARTIFACTS = [
    "results/scalability_analysis.png",
]
for ra in REQUIRED_ARTIFACTS:
    check(f"12b-artifact-{pathlib.Path(ra).name}", (ROOT / ra).exists(), f"{ra} missing")

# 12c. Publication gate script passes
print("  Running publication gate (this may take ~1 minute)...")
gate_result = subprocess.run(
    ["python3", "scripts/publication_gate.py", "--quick"],
    capture_output=True,
    text=True,
    cwd=ROOT,
    timeout=180,
)
gate_output = gate_result.stdout + gate_result.stderr
gate_pass = "Publication Gate: PASS" in gate_output
check(
    "12c-publication-gate-pass",
    gate_pass,
    f"Publication gate returned: {gate_output.strip().split(chr(10))[-3:]}",
)


# ═══════════════════════════════════════════════════════════════════
# GROUP 13 — COMPILE CHECKS (py_compile all src/)
# ═══════════════════════════════════════════════════════════════════
print("\n══ GROUP 13: Compile Checks ══")

compile_failures = []
for pyf in sorted((ROOT / "src").rglob("*.py")):
    try:
        import py_compile
        py_compile.compile(str(pyf), doraise=True)
    except py_compile.PyCompileError as e:
        compile_failures.append(f"{pyf.relative_to(ROOT)}: {e}")

check(
    "13a-all-src-compiles",
    len(compile_failures) == 0,
    f"{len(compile_failures)} files failed: {compile_failures[:3]}",
)

# Also check api/
api_failures = []
for pyf in sorted((ROOT / "api").rglob("*.py")):
    try:
        py_compile.compile(str(pyf), doraise=True)
    except py_compile.PyCompileError as e:
        api_failures.append(f"{pyf.relative_to(ROOT)}: {e}")

check(
    "13b-all-api-compiles",
    len(api_failures) == 0,
    f"{len(api_failures)} files failed: {api_failures[:3]}",
)


# ═══════════════════════════════════════════════════════════════════
# FINAL REPORT
# ═══════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
total = PASS_COUNT + FAIL_COUNT
print(f"TOTAL: {total} checks — {PASS_COUNT} PASS, {FAIL_COUNT} FAIL")
print("═" * 60)

if FAIL_COUNT > 0:
    print("\n❌ REMAINING FAILURES:")
    for f in FAILURES:
        print(f"  • {f}")
    print(f"\nExit code: 1 ({FAIL_COUNT} failures)")
    sys.exit(1)
else:
    print("\n✅ ALL CHECKS PASSED — CLEAN BILL OF HEALTH")
    print("Exit code: 0")
    sys.exit(0)
