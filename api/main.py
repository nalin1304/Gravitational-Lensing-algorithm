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
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
import numpy as np
import torch
import io
import base64
import logging
from datetime import datetime
import uuid
import asyncio
from pathlib import Path

# Import from src (shared utilities), NOT from app (frontend)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.utils.common import (
    load_pretrained_model,
    prepare_model_input,
    compute_classification_entropy
)
from src.ml.generate_dataset import generate_convergence_map_vectorized as generate_synthetic_convergence
from src.api_utils.auth import get_current_user, get_optional_user, create_access_token

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import Phase 12 database and routes
try:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from database import init_db, check_db_connection, get_db_info
    from api.auth_routes import router as auth_router
    from api.analysis_routes import router as analysis_router
    PHASE_12_ENABLED = True
except ImportError as e:
    logger.warning(f"Phase 12 features not available: {e}")
    PHASE_12_ENABLED = False

# Initialize FastAPI app
app = FastAPI(
    title="Gravitational Lensing API",
    description="REST API for gravitational lensing analysis using Physics-Informed Neural Networks",
    version="2.0.0",  # Phase 12
    docs_url="/docs",
    redoc_url="/redoc"
)

# Include Phase 12 routers
if PHASE_12_ENABLED:
    app.include_router(auth_router)
    app.include_router(analysis_router)
    logger.info("Phase 12 features enabled: Authentication and Database")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer(auto_error=False)

# Global model cache
MODEL_CACHE = {}

# Job tracking for background tasks
JOBS = {}


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
    
    @validator('profile_type')
    def validate_profile_type(cls, v):
        if v not in ["NFW", "Elliptical NFW"]:
            raise ValueError("profile_type must be 'NFW' or 'Elliptical NFW'")
        return v
    
    @validator('grid_size')
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


class InferenceResponse(BaseModel):
    """Response for model inference"""
    job_id: str
    predictions: Dict[str, float]
    uncertainties: Optional[Dict[str, float]] = None
    classification: Dict[str, float]
    entropy: float
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
    return datetime.utcnow().isoformat() + "Z"


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


# Note: Real authentication is now in src.api_utils.auth
# Use get_current_user for required auth, get_optional_user for optional auth


def load_model_cached():
    """Load model with caching"""
    if 'model' not in MODEL_CACHE:
        logger.info("Loading PINN model into cache...")
        MODEL_CACHE['model'] = load_pretrained_model()
        logger.info("Model loaded successfully")
    return MODEL_CACHE['model']


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Gravitational Lensing API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    
    Returns system status, GPU availability, database status, and timestamp
    """
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
    username: Optional[str] = Depends(get_optional_user)
):
    """
    Generate synthetic convergence map
    (Optional authentication - works for both authenticated and anonymous users)
    
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
        
        logger.info(f"Job {job_id}: Successfully generated convergence map")
        return response
        
    except Exception as e:
        logger.error(f"Job {job_id}: Error generating convergence map: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating convergence map: {str(e)}"
        )


@app.post("/api/v1/inference", response_model=InferenceResponse)
async def run_inference(
    request: InferenceRequest,
    username: Optional[str] = Depends(get_optional_user)
):
    """
    Run PINN model inference on convergence map
    (Optional authentication)
    
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
    
    try:
        # Load model
        model = load_model_cached()
        model.eval()
        
        # Prepare input
        convergence_map = np.array(request.convergence_map)
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
                timestamp=get_current_timestamp()
            )
        
        logger.info(f"Job {job_id}: Inference completed successfully")
        return response
        
    except Exception as e:
        logger.error(f"Job {job_id}: Error during inference: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error during inference: {str(e)}"
        )


@app.post("/api/v1/batch", response_model=Dict[str, str])
async def submit_batch_job(
    request: BatchJobRequest,
    background_tasks: BackgroundTasks,
    username: str = Depends(get_current_user)  # Batch jobs require authentication
):
    """
    Submit batch processing job
    (Requires authentication)
    
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
    return {
        "models": [
            {
                "name": "PINN",
                "version": "1.0.0",
                "description": "Physics-Informed Neural Network for lensing analysis",
                "input_size": [64, 64],
                "output_parameters": ["M_vir", "r_s", "ellipticity"],
                "loaded": "model" in MODEL_CACHE
            }
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
        "model_loaded": "model" in MODEL_CACHE,
        "gpu_available": torch.cuda.is_available(),
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
        for i, job_id in enumerate(job_ids):
            # Simulate processing (replace with actual logic)
            await asyncio.sleep(0.1)
            
            JOBS[batch_id]["completed"] = i + 1
            JOBS[batch_id]["progress"] = (i + 1) / len(job_ids) * 100
        
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
            "timestamp": get_current_timestamp()
        }
    )


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
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Actions to perform on startup"""
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


@app.on_event("shutdown")
async def shutdown_event():
    """Actions to perform on shutdown"""
    logger.info("Shutting down Gravitational Lensing API...")
    
    # Clear model cache
    MODEL_CACHE.clear()
    
    logger.info("Shutdown complete")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
