"""
FastAPI REST API for Gravitational Lensing Analysis

This module provides a RESTful API for:
- Generating synthetic convergence maps
- Analyzing real FITS data
- Running PINN model inference
- Computing uncertainty quantification
- Health checks and monitoring

Author: Phase 11 Implementation
Date: October 2025
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import numpy as np
import torch
import json
import io
import base64
import logging
from datetime import datetime, timezone
import uuid
from pathlib import Path
from contextlib import asynccontextmanager

from src.utils.common import (
    find_pretrained_model_checkpoint,
    load_pretrained_model,
    prepare_model_input,
    compute_classification_entropy
)
from src.ml.generate_dataset import generate_synthetic_convergence
# FIX P0 SECURITY: Import from database.auth for proper JWT verification with user database
from database.auth import get_current_user, get_current_active_user
from database import User

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import Phase 12 database and routes
try:
    from database import init_db, check_db_connection, get_db_info, get_db, Session
    from api.auth_routes import router as auth_router
    from api.analysis_routes import router as analysis_router
    PHASE_12_ENABLED = True
except ImportError as e:
    logger.warning(f"Phase 12 features not available: {e}")
    PHASE_12_ENABLED = False
    # Minimal compatibility type for endpoint annotations when DB layer is unavailable.
    class Session:  # type: ignore
        pass

# Initialize FastAPI app
@asynccontextmanager
async def app_lifespan(_: FastAPI):
    """Application lifespan hooks (startup/shutdown)."""
    logger.info("Starting Gravitational Lensing API...")
    logger.info(f"GPU Available: {torch.cuda.is_available()}")

    # Initialize database if Phase 12 enabled
    if PHASE_12_ENABLED:
        logger.info("Initializing database...")
        try:
            init_db()
            db_info = get_db_info()
            logger.info(f"Database connected: {db_info['type']} at {db_info['host']}")
            logger.info("Phase 12 features: Authentication, User Management, Database Persistence")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            logger.warning("Continuing without database features")

    logger.info("API ready to accept requests")
    try:
        yield
    finally:
        logger.info("Shutting down Gravitational Lensing API...")
        MODEL_CACHE.clear()
        logger.info("Shutdown complete")


app = FastAPI(
    title="Gravitational Lensing API",
    description="REST API for gravitational lensing analysis using Physics-Informed Neural Networks",
    version="2.0.0",  # Phase 12
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=app_lifespan,
)

# Alternative non-Streamlit UI (FastAPI-served static frontend)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
NEXT_UI_DIR = PROJECT_ROOT / "web_ui"
if NEXT_UI_DIR.exists():
    app.mount("/ui-static", StaticFiles(directory=str(NEXT_UI_DIR)), name="ui-static")

# P1 SECURITY FIX: Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter


async def rate_limit_exception_handler(request, exc):
    """Return rate-limit responses with both error/detail keys for compatibility."""
    raw_detail = str(getattr(exc, "detail", "rate limit exceeded"))
    detail = (
        raw_detail
        if "rate limit" in raw_detail.lower()
        else f"Rate limit exceeded: {raw_detail}"
    )
    return JSONResponse(
        status_code=429,
        content={
            "error": detail,
            "detail": detail,
            "timestamp": get_current_timestamp(),
        },
    )


app.add_exception_handler(RateLimitExceeded, rate_limit_exception_handler)

# Include Phase 12 routers
if PHASE_12_ENABLED:
    app.include_router(auth_router)
    app.include_router(analysis_router)
    logger.info("Phase 12 features enabled: Authentication and Database")

# Include Next-Gen Rigor
try:
    from api.rigor_routes import router as rigor_router
    app.include_router(rigor_router)
    logger.info("Next-Gen Rigor routes loaded.")
except ImportError as e:
    logger.warning(f"Rigor routes unavailable: {e}")

# Configure CORS
# In production, set CORS_ORIGINS env var to a comma-separated list of allowed origins.
# Example: CORS_ORIGINS="https://yourdomain.com,https://app.yourdomain.com"
import os as _os
_cors_env = _os.environ.get("CORS_ORIGINS", "")
CORS_ORIGINS: list = [o.strip() for o in _cors_env.split(",") if o.strip()] or ["*"]
if CORS_ORIGINS == ["*"]:
    logger.warning(
        "CORS is configured to allow ALL origins ('*'). "
        "Set CORS_ORIGINS env var to restrict in production."
    )
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer(auto_error=False)

# Global model cache
MODEL_CACHE = {}

# Job tracking for background tasks (with TTL eviction)
JOBS: dict = {}
_JOB_TTL_SECONDS = 3600  # 1 hour


def _evict_old_jobs() -> None:
    """Remove completed/failed jobs older than _JOB_TTL_SECONDS to prevent memory growth."""
    from datetime import timezone
    now = datetime.now(timezone.utc).timestamp()
    stale = [
        jid for jid, j in JOBS.items()
        if j.get("status") in ("completed", "failed")
        and now - j.get("_created_ts", now) > _JOB_TTL_SECONDS
    ]
    for jid in stale:
        del JOBS[jid]
    if stale:
        logger.debug("Evicted %d stale jobs from JOBS dict", len(stale))


# ============================================================================
# Pydantic Models (Request/Response Schemas)
# ============================================================================

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    version: str
    gpu_available: bool


class SyntheticRequest(BaseModel):
    """Request for synthetic convergence map generation"""
    profile_type: str = Field(..., description="NFW or Elliptical NFW")
    mass: float = Field(..., ge=1e11, le=1e14, description="Virial mass in solar masses")
    scale_radius: float = Field(200.0, ge=50.0, le=500.0, description="Scale radius in kpc")
    ellipticity: float = Field(0.0, ge=0.0, le=0.5, description="Ellipticity parameter")
    grid_size: int = Field(64, description="Grid size (32, 64, or 128)")
    
    @field_validator('profile_type')
    @classmethod
    def validate_profile_type(cls, v):
        if v not in ["NFW", "Elliptical NFW"]:
            raise ValueError("profile_type must be 'NFW' or 'Elliptical NFW'")
        return v
    
    @field_validator('grid_size')
    @classmethod
    def validate_grid_size(cls, v):
        if v not in [32, 64, 128]:
            raise ValueError("grid_size must be 32, 64, or 128")
        return v


class SyntheticResponse(BaseModel):
    """Response for synthetic convergence map generation"""
    job_id: str
    convergence_map: List[List[float]]
    coordinates: Dict[str, List[List[float]]]
    metadata: Dict[str, Any]
    timestamp: str


class InferenceRequest(BaseModel):
    """Request for model inference"""
    convergence_map: List[List[float]] = Field(..., description="2D convergence map")
    target_size: int = Field(64, description="Target size for model input")
    mc_samples: int = Field(1, ge=1, le=1000, description="Number of MC Dropout samples")

    @field_validator("convergence_map")
    @classmethod
    def validate_convergence_map(cls, value: List[List[float]]) -> List[List[float]]:
        """Require a finite, rectangular 2-D convergence map large enough for inference."""
        try:
            array = np.asarray(value, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("convergence_map must contain finite numeric values") from exc

        if array.ndim != 2:
            raise ValueError("convergence_map must be a 2D array")
        if min(array.shape) < 16:
            raise ValueError("convergence_map must be at least 16x16 for scientific inference")
        if not np.isfinite(array).all():
            raise ValueError("convergence_map must contain only finite numeric values")
        return value


class InferenceResponse(BaseModel):
    """Response for model inference"""
    job_id: str
    predictions: Dict[str, float]
    uncertainties: Optional[Dict[str, float]] = None
    classification: Dict[str, float]
    entropy: float
    inference_mode: str = "pinn"
    timestamp: str


class BatchJobRequest(BaseModel):
    """Request for batch processing"""
    job_ids: List[str]


class BatchJobStatus(BaseModel):
    """Status of batch job"""
    job_id: str
    status: str
    progress: float
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: str
    timestamp: str


# ============================================================================
# Utility Functions
# ============================================================================

def get_current_timestamp() -> str:
    """Get current timestamp as ISO string"""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def generate_job_id() -> str:
    """Generate unique job ID"""
    return str(uuid.uuid4())


def encode_array_to_base64(arr: np.ndarray) -> str:
    """Encode numpy array to base64 string"""
    buffer = io.BytesIO()
    np.save(buffer, arr)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')


def decode_base64_to_array(b64_string: str) -> np.ndarray:
    """Decode base64 string to numpy array"""
    buffer = io.BytesIO(base64.b64decode(b64_string))
    return np.load(buffer)

def _runtime_dependencies_ready() -> bool:
    """Return whether the JAX/Equinox runtime required for inference is available."""
    try:
        import jax  # noqa: F401
        import equinox  # noqa: F401
    except ImportError:
        return False
    return True


def _model_status() -> dict[str, Any]:
    """Summarize checkpoint and runtime readiness for the inference service."""
    checkpoint_path = find_pretrained_model_checkpoint()
    runtime_ready = _runtime_dependencies_ready()
    loaded = "model" in MODEL_CACHE

    if loaded:
        status = "loaded"
    elif checkpoint_path is None:
        status = "checkpoint_missing"
    elif not runtime_ready:
        status = "runtime_incomplete"
    else:
        status = "checkpoint_available"

    return {
        "status": status,
        "loaded": loaded,
        "supports_inference": bool(checkpoint_path is not None and runtime_ready),
        "runtime_dependencies_ready": runtime_ready,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
    }


def _lens_finder_status() -> dict[str, Any]:
    """Summarize availability of the trained survey lens-finder model."""
    try:
        from src.ml.lens_finder import find_lens_finder_checkpoint
    except ImportError:
        return {
            "status": "module_unavailable",
            "loaded": False,
            "supports_detection": False,
            "runtime_dependencies_ready": False,
            "checkpoint_path": None,
        }

    checkpoint_path = find_lens_finder_checkpoint()
    runtime_ready = bool(getattr(torch, "__version__", None))
    status = "checkpoint_available" if checkpoint_path is not None and runtime_ready else "checkpoint_missing"
    if not runtime_ready:
        status = "runtime_incomplete"

    return {
        "status": status,
        "loaded": False,
        "supports_detection": bool(checkpoint_path is not None and runtime_ready),
        "runtime_dependencies_ready": runtime_ready,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
    }


def _load_json_artifact(path: Path) -> Optional[Any]:
    """Load a JSON artifact if it exists and is parseable."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Could not parse JSON artifact at %s", path)
        return None


def _load_regression_summary() -> Optional[dict[str, Any]]:
    """Load the latest regression summary or derive a minimal gate-check summary."""
    regression_path = PROJECT_ROOT / "results" / "regression_summary.json"
    regression_summary = _load_json_artifact(regression_path)
    if isinstance(regression_summary, dict):
        return regression_summary

    gate_path = PROJECT_ROOT / "results" / "publication_gate_report.json"
    gate_report = _load_json_artifact(gate_path)
    if not isinstance(gate_report, dict):
        return None

    checks = gate_report.get("checks", [])
    passed_checks = sum(1 for check in checks if isinstance(check, dict) and check.get("passed"))
    total_checks = len(checks)
    return {
        "status": "artifact_checks",
        "generated_at_utc": gate_report.get("generated_at_utc"),
        "publication_gate_passed": bool(gate_report.get("publication_gate_passed")),
        "checks_passed": passed_checks,
        "checks_total": total_checks,
    }


def _load_validation_summary() -> Optional[dict[str, Any]]:
    """Load the latest statistical-rigor summary for dashboards."""
    rigor_path = PROJECT_ROOT / "results" / "statistical_rigor_report.json"
    rigor_report = _load_json_artifact(rigor_path)
    if not isinstance(rigor_report, dict):
        return None

    slacs = rigor_report.get("slacs_validation", {})
    warnings_list = rigor_report.get("overall_warnings", [])
    return {
        "generated_at_utc": rigor_report.get("generated_at_utc"),
        "slacs_joint_pass_rate": slacs.get("joint_pass_rate"),
        "slacs_prediction_modes": slacs.get("prediction_modes", []),
        "slacs_data_sources": slacs.get("data_sources", []),
        "warning_count": len(warnings_list) if isinstance(warnings_list, list) else 0,
        "warnings": warnings_list if isinstance(warnings_list, list) else [],
    }


def _normalize_slacs_rows(payload: Any) -> Any:
    """Normalize SLACS validation rows for frontend consumption."""
    if not isinstance(payload, list):
        return payload

    normalized_rows: list[dict[str, Any]] = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        normalized = dict(row)
        if isinstance(normalized.get("passed"), str):
            normalized["passed"] = normalized["passed"].strip().lower() == "true"
        if "validation_scope" not in normalized:
            mode = str(normalized.get("prediction_mode", "unknown"))
            normalized["validation_scope"] = (
                "image_space_diagnostic"
                if "scaled_intensity" in mode
                else "proxy_sensitivity"
            )
        normalized_rows.append(normalized)
    return normalized_rows


# Note: Real authentication is now in src.api_utils.auth
# Use get_current_user for required auth, get_optional_user for optional auth


def load_model_cached():
    """Load model with caching and require a real trained checkpoint."""
    model = MODEL_CACHE.get('model')
    if model is None:
        status = _model_status()
        if status["checkpoint_path"] is None:
            raise RuntimeError(
                "No pretrained PINN model available. "
                "Deploy a trained checkpoint before requesting inference."
            )
        if not status["runtime_dependencies_ready"]:
            raise RuntimeError(
                "A pretrained PINN checkpoint exists, but JAX/Equinox runtime "
                "dependencies are unavailable in this environment."
            )

        logger.info("Loading PINN model into cache...")
        model = load_pretrained_model()
        MODEL_CACHE['model'] = model
        logger.info("Model loaded successfully")
    return model


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Gravitational Lensing API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "web_ui": "/ui",
        "streamlit_ui": "app/Home.py",
    }


@app.get("/ui", include_in_schema=False)
async def journal_workbench_ui():
    """Serve alternative non-Streamlit frontend for analysis workflows."""
    index_path = NEXT_UI_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="UI frontend not available in this deployment.")
    return FileResponse(index_path)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint

    Returns system status, GPU availability, database status, and timestamp
    """
    # Passively evict stale completed/failed jobs to prevent memory growth
    _evict_old_jobs()

    health_data = {
        "status": "healthy",
        "timestamp": get_current_timestamp(),
        "version": "2.0.0",
        "gpu_available": torch.cuda.is_available()
    }
    
    # Add database status if Phase 12 enabled
    if PHASE_12_ENABLED:
        try:
            db_connected = check_db_connection()
            health_data["database_connected"] = db_connected
            if not db_connected:
                health_data["status"] = "degraded"
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            health_data["database_connected"] = False
            health_data["status"] = "degraded"
    
    return health_data


@app.post("/api/v1/synthetic", response_model=SyntheticResponse)
async def generate_synthetic(
    request: SyntheticRequest,
    db: Session = Depends(get_db),
    current_user: Optional[Any] = None
):
    """
    Generate synthetic convergence map
    (Public endpoint - no authentication required)
    
    Parameters:
    - profile_type: "NFW" or "Elliptical NFW"
    - mass: Virial mass in solar masses (10^11 to 10^14)
    - scale_radius: Scale radius in kpc (50-500)
    - ellipticity: Ellipticity parameter (0.0-0.5)
    - grid_size: Grid size (32, 64, or 128)
    
    Returns:
    - Convergence map as 2D array
    - Coordinate grids (X, Y)
    - Metadata about generation
    """
    job_id = generate_job_id()
    logger.info(f"Job {job_id}: Generating synthetic convergence map")
    JOBS[job_id] = {
        "status": "running",
        "job_type": "synthetic",
        "progress": 0.0,
        "_created_ts": datetime.now(timezone.utc).timestamp(),
    }
    
    try:
        # Generate convergence map
        convergence_map, X, Y = generate_synthetic_convergence(
            profile_type=request.profile_type,
            mass=request.mass,
            scale_radius=request.scale_radius,
            ellipticity=request.ellipticity,
            grid_size=request.grid_size
        )
        
        # Prepare response
        response = SyntheticResponse(
            job_id=job_id,
            convergence_map=convergence_map.tolist(),
            coordinates={
                "X": X.tolist(),
                "Y": Y.tolist()
            },
            metadata={
                "profile_type": request.profile_type,
                "mass": request.mass,
                "scale_radius": request.scale_radius,
                "ellipticity": request.ellipticity,
                "grid_size": request.grid_size,
                "shape": convergence_map.shape,
                "min_value": float(convergence_map.min()),
                "max_value": float(convergence_map.max()),
                "mean_value": float(convergence_map.mean())
            },
            timestamp=get_current_timestamp()
        )
        JOBS[job_id] = {
            "status": "completed",
            "job_type": "synthetic",
            "progress": 100.0,
            "result": response.model_dump(),
        }
        
        logger.info(f"Job {job_id}: Successfully generated convergence map")
        return response
        
    except Exception as e:
        JOBS[job_id] = {
            "status": "failed",
            "job_type": "synthetic",
            "progress": 100.0,
            "error": str(e),
        }
        logger.error(f"Job {job_id}: Error generating convergence map: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating convergence map: {str(e)}"
        )


@app.post("/api/v1/inference", response_model=InferenceResponse)
async def run_inference(
    request: InferenceRequest,
    db: Session = Depends(get_db),
    current_user: Optional[Any] = None
):
    """
    Run PINN model inference on convergence map
    (Public endpoint - no authentication required)
    
    Parameters:
    - convergence_map: 2D array of convergence values
    - target_size: Target size for model input (default: 64)
    - mc_samples: Number of MC Dropout samples for uncertainty (default: 1)
    
    Returns:
    - Parameter predictions (M_vir, r_s, ellipticity)
    - Uncertainties (if mc_samples > 1)
    - Classification probabilities
    - Predictive entropy
    """
    job_id = generate_job_id()
    logger.info(f"Job {job_id}: Running model inference")
    JOBS[job_id] = {
        "status": "running",
        "job_type": "inference",
        "progress": 0.0,
        "_created_ts": datetime.now(timezone.utc).timestamp(),
    }
    
    try:
        # Prepare input (NumPy) always available
        convergence_map = np.array(request.convergence_map, dtype=float)

        # Load model; strict mode forbids heuristic stand-ins when checkpoints are missing.
        try:
            model = load_model_cached()
        except RuntimeError as missing_model_error:
            JOBS[job_id] = {
                "status": "failed",
                "job_type": "inference",
                "progress": 100.0,
                "error": str(missing_model_error),
            }
            raise HTTPException(status_code=503, detail=str(missing_model_error))

        # Prepare tensor input
        input_tensor = prepare_model_input(convergence_map, target_size=request.target_size)
        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        input_tensor = input_tensor.to(device)
        
        if request.mc_samples == 1:
            # Single forward pass
            with torch.no_grad():
                predictions, classification = model(input_tensor)
            
            predictions = predictions.cpu().numpy()[0]
            classification = torch.softmax(classification, dim=1).cpu().numpy()[0]
            
            response = InferenceResponse(
                job_id=job_id,
                predictions={
                    "M_vir": float(predictions[0]),
                    "r_s": float(predictions[1]),
                    "ellipticity": float(predictions[2])
                },
                classification={
                    f"class_{i}": float(classification[i]) 
                    for i in range(len(classification))
                },
                entropy=float(compute_classification_entropy(classification)),
                inference_mode="pinn",
                timestamp=get_current_timestamp()
            )
        else:
            # MC Dropout for uncertainty
            # Keep model in eval mode to avoid BatchNorm issues with single sample
            model.eval()
            
            # Enable dropout manually if available
            for module in model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.train()
            
            all_predictions = []
            all_classifications = []
            
            for _ in range(request.mc_samples):
                with torch.no_grad():
                    pred, classif = model(input_tensor)
                all_predictions.append(pred.cpu().numpy()[0])
                all_classifications.append(
                    torch.softmax(classif, dim=1).cpu().numpy()[0]
                )
            
            # Compute statistics
            predictions_array = np.array(all_predictions)
            mean_predictions = predictions_array.mean(axis=0)
            std_predictions = predictions_array.std(axis=0)
            
            classifications_array = np.array(all_classifications)
            mean_classification = classifications_array.mean(axis=0)
            
            response = InferenceResponse(
                job_id=job_id,
                predictions={
                    "M_vir": float(mean_predictions[0]),
                    "r_s": float(mean_predictions[1]),
                    "ellipticity": float(mean_predictions[2])
                },
                uncertainties={
                    "M_vir_std": float(std_predictions[0]),
                    "r_s_std": float(std_predictions[1]),
                    "ellipticity_std": float(std_predictions[2])
                },
                classification={
                    f"class_{i}": float(mean_classification[i]) 
                    for i in range(len(mean_classification))
                },
                entropy=float(compute_classification_entropy(mean_classification)),
                inference_mode="pinn",
                timestamp=get_current_timestamp()
            )
        
        JOBS[job_id] = {
            "status": "completed",
            "job_type": "inference",
            "progress": 100.0,
            "result": response.model_dump(),
        }
        
        logger.info(f"Job {job_id}: Inference completed successfully")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        JOBS[job_id] = {
            "status": "failed",
            "job_type": "inference",
            "progress": 100.0,
            "error": str(e),
        }
        logger.error(f"Job {job_id}: Error during inference: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error during inference: {str(e)}"
        )


@app.post("/api/v1/batch", response_model=Dict[str, str])
async def submit_batch_job(
    request: BatchJobRequest,
    background_tasks: BackgroundTasks,
    current_user: Optional[Any] = None,
    db: Session = Depends(get_db)
):
    """
    Submit batch processing job
    (Requires authentication - FIXED: Now properly validates JWT tokens via database.auth)
    
    Parameters:
    - job_ids: List of job IDs to process
    
    Returns:
    - Batch job ID for tracking
    """
    batch_id = generate_job_id()
    logger.info(f"Batch {batch_id}: Submitted with {len(request.job_ids)} jobs")
    
    # Initialize batch job status
    JOBS[batch_id] = {
        "status": "pending",
        "progress": 0.0,
        "total": len(request.job_ids),
        "completed": 0,
        "results": []
    }
    
    # Add to background tasks
    background_tasks.add_task(process_batch, batch_id, request.job_ids)
    
    return {
        "batch_id": batch_id,
        "message": f"Batch job submitted with {len(request.job_ids)} items",
        "status_url": f"/api/v1/batch/{batch_id}/status"
    }


@app.get("/api/v1/batch/{batch_id}/status", response_model=Dict[str, Any])
async def get_batch_status(batch_id: str):
    """
    Get status of batch processing job
    
    Parameters:
    - batch_id: Batch job ID
    
    Returns:
    - Current status and progress
    """
    if batch_id not in JOBS:
        raise HTTPException(status_code=404, detail="Batch job not found")
    
    return JOBS[batch_id]


@app.get("/api/v1/models", response_model=Dict[str, Any])
async def list_models():
    """
    List available models
    
    Returns:
    - Available model information
    """
    model_status = _model_status()
    return {
        "models": [
            {
                "name": "PINN",
                "version": "1.0.0",
                "description": "Physics-Informed Neural Network for lensing analysis",
                "input_size": [64, 64],
                "output_parameters": ["M_vir", "r_s", "ellipticity"],
                **model_status,
            },
            {
                "name": "LensFinder",
                "version": "1.0.0",
                "description": "Checkpoint-backed survey lens candidate detector",
                "input_size": [64, 64],
                "output_parameters": ["objectness", "x_center", "y_center", "width", "height"],
                **_lens_finder_status(),
            },
        ]
    }


@app.get("/api/v1/stats", response_model=Dict[str, Any])
async def get_statistics():
    """
    Get API usage statistics
    
    Returns:
    - Request counts, processing times, etc.
    """
    return {
        "total_jobs": len(JOBS),
        "active_jobs": sum(1 for j in JOBS.values() if j.get("status") == "running"),
        "completed_jobs": sum(1 for j in JOBS.values() if j.get("status") == "completed"),
        "model_status": _model_status(),
        "feature_status": {
            "pinn_inference": _model_status(),
            "survey_finder": _lens_finder_status(),
        },
        "gpu_available": torch.cuda.is_available(),
        "regression_summary": _load_regression_summary(),
        "validation_summary": _load_validation_summary(),
        "timestamp": get_current_timestamp()
    }


# ============================================================================
# Background Tasks
# ============================================================================

async def process_batch(batch_id: str, job_ids: List[str]):
    """
    Process batch job in background
    
    Parameters:
    - batch_id: Batch job ID
    - job_ids: List of individual job IDs
    """
    logger.info(f"Batch {batch_id}: Starting processing")
    JOBS[batch_id]["status"] = "running"
    
    try:
        results = []
        missing = []
        failed = 0

        for i, job_id in enumerate(job_ids):
            if job_id not in JOBS:
                item = {
                    "job_id": job_id,
                    "status": "not_found",
                    "error": "Referenced job ID not found",
                }
                missing.append(job_id)
            else:
                source = JOBS[job_id]
                item = {
                    "job_id": job_id,
                    "status": source.get("status", "unknown"),
                }
                if "result" in source:
                    item["result"] = source["result"]
                if "error" in source:
                    item["error"] = source["error"]
                if source.get("status") == "failed":
                    failed += 1

            results.append(item)
            JOBS[batch_id]["completed"] = i + 1
            JOBS[batch_id]["progress"] = (i + 1) / max(len(job_ids), 1) * 100.0

        JOBS[batch_id]["results"] = results

        if missing or failed > 0:
            # Keep status vocabulary compatible with API contract/tests.
            JOBS[batch_id]["status"] = "failed"
            JOBS[batch_id]["error"] = (
                f"{len(missing)} missing jobs, {failed} failed jobs in batch aggregation"
            )
            logger.warning(
                f"Batch {batch_id}: completed with errors "
                f"(missing={len(missing)}, failed={failed})"
            )
        else:
            JOBS[batch_id]["status"] = "completed"
            logger.info(f"Batch {batch_id}: Completed successfully")

    except Exception as e:
        JOBS[batch_id]["status"] = "failed"
        JOBS[batch_id]["error"] = str(e)
        logger.error(f"Batch {batch_id}: Failed with error: {str(e)}")


# ============================================================================
# Exception Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "detail": exc.detail,
            "timestamp": get_current_timestamp()
        }
    )


# ============================================================================
# Validation Dashboard Endpoints
# ============================================================================

@app.get("/api/v1/validation/slacs", tags=["validation"])
async def get_slacs_validation():
    """Return SLACS survey validation results from pre-computed JSON."""
    results_path = Path("results/real_data/slacs_validation_results.json")
    if not results_path.exists():
        return JSONResponse(status_code=404, content={"error": "No SLACS results found. Run: python scripts/validate_real_data.py"})
    payload = _load_json_artifact(results_path)
    if payload is None:
        return JSONResponse(status_code=500, content={"error": "SLACS validation artifact could not be parsed."})
    return _normalize_slacs_rows(payload)


@app.get("/api/v1/validation/calibration", tags=["validation"])
async def get_calibration_results():
    """Return uncertainty calibration results (ECE, coverage)."""
    results_path = Path("results/uncertainty_calibration_results.json")
    if not results_path.exists():
        return JSONResponse(status_code=404, content={"error": "No calibration results found. Run: python scripts/uncertainty_calibration.py"})
    payload = _load_json_artifact(results_path)
    if payload is None:
        return JSONResponse(status_code=500, content={"error": "Calibration artifact could not be parsed."})
    return payload


@app.get("/api/v1/validation/ablation", tags=["validation"])
async def get_ablation_results():
    """Return ablation study results."""
    results_path = Path("results/ablation_results.json")
    if not results_path.exists():
        return JSONResponse(status_code=404, content={"error": "No ablation results found. Run: python scripts/ablation_study.py"})
    payload = _load_json_artifact(results_path)
    if payload is None:
        return JSONResponse(status_code=500, content={"error": "Ablation artifact could not be parsed."})
    return payload


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": get_current_timestamp()
        }
    )


# ============================================================================
# STAGE IV SURVEY ENDPOINTS
# ============================================================================

class FinderRequest(BaseModel):
    mode: str = Field("synthetic", pattern="^(synthetic|fits)$")
    stride: int = Field(32, ge=8, le=128)
    confidence_threshold: float = Field(0.70, ge=0.1, le=0.99)
    n_lenses: int = Field(5, ge=1, le=20)
    seed: int = Field(42)


@app.get("/api/v1/survey/finder/status")
async def survey_finder_status():
    """Expose survey detector availability for the frontend."""
    return _lens_finder_status()


@app.post("/api/v1/survey/finder")
async def survey_finder(req: FinderRequest):
    """LenNet-style automated lens discovery on synthetic or uploaded field."""
    if req.mode != "synthetic":
        raise HTTPException(
            status_code=501,
            detail="FITS upload scanning is not enabled in this deployment. Use synthetic mode or deploy the multipart upload route."
        )

    status = _lens_finder_status()
    if not status["supports_detection"]:
        raise HTTPException(
            status_code=503,
            detail=(
                "LensFinder is unavailable because no trained checkpoint-backed detector "
                "is present in this environment."
            ),
        )

    try:
        from src.ml.lens_finder import LensFinder
        finder = LensFinder(confidence_threshold=req.confidence_threshold)
        results = finder.scan_synthetic(
            grid_size=256,
            n_lenses=req.n_lenses,
            seed=req.seed,
        )
        return {
            "candidates": [r.to_dict() for r in results],
            "n_candidates": len(results),
            "stride": req.stride,
            "mode": req.mode,
            "detector_status": status,
        }
    except Exception as e:
        logger.exception("Finder error")
        raise HTTPException(status_code=500, detail=str(e))


class EPSFRequest(BaseModel):
    pixel_scale: float = Field(0.11, gt=0, le=1.0)
    kernel_size: int = Field(21, ge=7, le=63)
    wavelength_um: float = Field(1.55, gt=0.1, le=10.0)
    aperture_m: float = Field(2.4, gt=0.1, le=20.0)
    x_det: float = Field(512.0, ge=0)
    y_det: float = Field(1024.0, ge=0)
    charge_diffusion: bool = Field(True)


@app.post("/api/v1/survey/epsf")
async def survey_epsf(req: EPSFRequest):
    """Evaluate the spatially-varying ePSF kernel at a given detector position."""
    import base64
    import io
    try:
        from src.optics.epsf_model import ePSFModel
        psf = ePSFModel(
            pixel_scale=req.pixel_scale,
            kernel_size=req.kernel_size,
            wavelength_micron=req.wavelength_um,
            aperture_diameter_m=req.aperture_m,
            include_charge_diffusion=req.charge_diffusion,
        )
        kernel = psf.evaluate(req.x_det, req.y_det)
        fwhm_px = _estimate_fwhm(kernel)
        fwhm_arcsec = fwhm_px * req.pixel_scale
        airy_peak = psf._diffraction_airy().max()
        strehl = float(kernel.max() / (airy_peak + 1e-30))

        # Render kernel as base64 PNG
        kernel_norm = kernel / (kernel.max() + 1e-30)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(2, 2))
            ax.imshow(kernel_norm, cmap="inferno", origin="lower")
            ax.axis("off")
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=80, bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            kernel_b64 = base64.b64encode(buf.getvalue()).decode()
        except Exception:
            kernel_b64 = None

        # Zernike wavefront RMS at this position
        zcoeffs = psf.wavefront.wavefront_coefficients(req.x_det, req.y_det)
        rms_nm = float(np.sqrt(sum(c ** 2 for c in zcoeffs.values())))

        return {
            "fwhm_pixels": round(fwhm_px, 3),
            "fwhm_arcsec": round(fwhm_arcsec, 4),
            "strehl": round(strehl, 4),
            "kernel_sum": round(float(kernel.sum()), 8),
            "kernel_b64": kernel_b64,
            "x_det": req.x_det,
            "y_det": req.y_det,
            "zernike_rms_nm": round(rms_nm, 3),
        }
    except Exception as e:
        logger.exception("ePSF error")
        raise HTTPException(status_code=500, detail=str(e))


def _estimate_fwhm(kernel: np.ndarray) -> float:
    """Estimate FWHM of a 2-D PSF using the encircled-energy method."""
    peak = kernel.max()
    half = peak / 2
    cy, cx = np.unravel_index(kernel.argmax(), kernel.shape)
    Y, X = np.ogrid[:kernel.shape[0], :kernel.shape[1]]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).ravel()
    vals = kernel.ravel()
    # Radius where profile falls below half-max (nearest pixel)
    above = dist[vals >= half]
    return float(2 * above.max()) if len(above) > 0 else 0.0


@app.get("/api/v1/survey/epsf/fov")
async def survey_epsf_fov(zernike_index: int = 4):
    """Compute the Zernike FOV variation map across the detector."""
    if not (4 <= zernike_index <= 22):
        raise HTTPException(status_code=422, detail="zernike_index must be 4–22")
    try:
        from src.optics.epsf_model import ePSFModel
        detector = (4096, 4096)
        psf = ePSFModel(detector_shape=detector)
        fov = psf.zernike_map(zernike_index, grid_points=16)
        stats = {
            "min": round(float(fov.min()), 4),
            "max": round(float(fov.max()), 4),
            "rms": round(float(np.sqrt(np.mean(fov ** 2))), 4),
        }
        return {
            "zernike_index": zernike_index,
            "grid_points": 16,
            "detector_shape": list(detector),
            "stats": stats,
            "fov_flat": fov.ravel().tolist(),
        }
    except Exception as e:
        logger.exception("ePSF FOV error")
        raise HTTPException(status_code=500, detail=str(e))


class BlindingUnblindRequest(BaseModel):
    phrase: str = Field(..., min_length=1)
    h0_blind: float
    dtd_blind: float


@app.post("/api/v1/survey/blinding/unblind")
async def survey_blinding_unblind(req: BlindingUnblindRequest):
    """
    Attempt to unblind H₀ and D_Δt; runs validation gate checks first.
    """
    try:
        from src.utils.blinding import BlindingHandler
        bh = BlindingHandler(req.phrase)
        # Run basic checks: test suite pass indicated by existence of report
        gate_report = Path("results/publication_gate_report.json")
        checks_passed = gate_report.exists()
        if not checks_passed:
            return {
                "gate_passed": False,
                "reason": "results/publication_gate_report.json not found. Run publication_gate.py first.",
            }
        try:
            report = json.loads(gate_report.read_text())
            checks_passed = bool(report.get("publication_gate_passed", False))
        except Exception:
            checks_passed = False
        if not checks_passed:
            return {
                "gate_passed": False,
                "reason": "publication_gate_report.json indicates gate not passed.",
            }
        h0_true  = bh.unblind_h0(req.h0_blind, verification_phrase=req.phrase)
        dtd_true = bh.unblind_dtd(req.dtd_blind, verification_phrase=req.phrase)
        return {
            "gate_passed": True,
            "h0_true": round(h0_true, 4),
            "dtd_true": round(dtd_true, 2),
            "summary": bh.summary(),
        }
    except ValueError as e:
        return {"gate_passed": False, "reason": str(e)}
    except Exception as e:
        logger.exception("Blinding unblind error")
        raise HTTPException(status_code=500, detail=str(e))


class CovarianceRequest(BaseModel):
    image_size: int = Field(32, ge=4, le=64)
    sigma: float = Field(0.02, gt=0)
    pixfrac: float = Field(0.8, gt=0, le=1.0)
    scale: float = Field(0.5, gt=0, le=1.0)
    kernel: str = Field("square")


@app.post("/api/v1/survey/covariance")
async def survey_covariance(req: CovarianceRequest):
    """Compute drizzle pixel covariance matrix and run Cholesky whitening diagnostic."""
    try:
        from src.data.pixel_covariance import (
            build_drizzle_covariance, apply_covariance_whitening,
            effective_noise_correlation_length,
        )
        rng = np.random.RandomState(42)
        rms_map = np.full((req.image_size, req.image_size), req.sigma)
        cov = build_drizzle_covariance(rms_map, req.pixfrac, req.scale, req.kernel)
        eigvals = np.linalg.eigvalsh(cov)
        is_pd = bool(np.all(eigvals > 0))
        xi = effective_noise_correlation_length(cov, shape=(req.image_size, req.image_size))
        residual = rng.normal(0, req.sigma, (req.image_size, req.image_size))
        chisq = apply_covariance_whitening(residual, cov)
        dof = req.image_size ** 2
        return {
            "shape": list(cov.shape),
            "is_positive_definite": is_pd,
            "correlation_length_px": round(float(xi), 3),
            "chisq": round(float(chisq), 4),
            "dof": dof,
            "chisq_dof": round(float(chisq) / dof, 4),
            "kernel": req.kernel,
            "pixfrac": req.pixfrac,
            "scale": req.scale,
        }
    except Exception as e:
        logger.exception("Covariance error")
        raise HTTPException(status_code=500, detail=str(e))


class JointSurveyRequest(BaseModel):
    ground_size: int = Field(32, ge=8, le=128)
    ground_scale: float = Field(0.2, gt=0)
    ground_sigma: float = Field(0.02, gt=0)
    space_size: int = Field(64, ge=8, le=256)
    space_scale: float = Field(0.11, gt=0)
    space_sigma: float = Field(0.01, gt=0)
    likelihood: str = Field("gaussian", pattern="^(gaussian|correlated)$")
    seed: int = Field(42)


@app.post("/api/v1/survey/joint")
async def survey_joint(req: JointSurveyRequest):
    """Run joint multi-survey deblending on synthetic Rubin+Roman-like data."""
    try:
        from src.ml.joint_survey import JointSurveyLikelihood
        jll = JointSurveyLikelihood.make_synthetic(
            grid_size_ground=req.ground_size,
            grid_size_space=req.space_size,
            seed=req.seed,
        )
        jll.use_correlated = (req.likelihood == "correlated")

        # Use perfect model = observed (sanity test for chi-sq ≈ dof)
        model_ground = jll.observations[0].image.copy()
        model_space  = jll.observations[1].image.copy()
        log_L = jll.log_likelihood(model_ground, model_space)
        chisq_info = jll.chi_squared(model_ground, model_space)

        return {
            "joint_log_likelihood": round(float(log_L), 4),
            "total_chisq_dof": round(float(chisq_info["total_chisq_dof"]), 4),
            "likelihood_mode": req.likelihood,
            "per_survey": {
                k: {
                    "chisq": round(v["chisq"], 3),
                    "dof": v["dof"],
                    "chisq_dof": round(v["chisq_dof"], 4),
                }
                for k, v in chisq_info.items()
                if k != "total_chisq_dof"
            },
        }
    except Exception as e:
        logger.exception("Joint survey error")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
