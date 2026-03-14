from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/rigor", tags=["rigor"])

class SBIRequest(BaseModel):
    convergence_map: list[list[float]] = Field(..., description="2D convergence map")
    n_samples: int = Field(1000, description="Number of posterior samples")

@router.post("/sbi")
async def run_sbi_inference(req: SBIRequest):
    try:
        from src.ml.sbi_npe import NeuralPosteriorEstimator
        import jax.numpy as jnp
        
        obs_arr = np.array(req.convergence_map, dtype=np.float32).flatten()
        obs_dim = obs_arr.shape[0]
        npe = NeuralPosteriorEstimator(param_dim=3, obs_dim=obs_dim)
        obs = jnp.array(obs_arr).reshape(1, obs_dim)
        samples = npe.sample(obs, num_samples=req.n_samples)
        perc = np.percentile(np.array(samples), [16, 50, 84], axis=0)
        
        return {
            "mode": "SBI NPE",
            "obs_dim": obs_dim,
            "M_vir": {"16th": float(perc[0, 0]), "median": float(perc[1, 0]), "84th": float(perc[2, 0])},
            "ratio": {"16th": float(perc[0, 1]), "median": float(perc[1, 1]), "84th": float(perc[2, 1])},
            "eff": {"16th": float(perc[0, 2]), "median": float(perc[1, 2]), "84th": float(perc[2, 2])},
        }
    except Exception as e:
        logger.exception("SBI Error")
        raise HTTPException(status_code=500, detail=str(e))

class StarletRequest(BaseModel):
    image: list[list[float]]
    n_scales: int = Field(3)
    lambda_reg: float = Field(0.01)
    max_iter: int = Field(10)

@router.post("/starlet")
async def run_starlet(req: StarletRequest):
    """Starlet sparse source reconstruction.
    
    Uses identity forward/adjoint operators by default — equivalent to 
    denoising the image in source plane (no lens model inversion).
    For full lens-model inversion, provide a lensing operator externally.
    """
    try:
        from src.ml.starlet_reconstruction import StarletTransform
        st = StarletTransform(num_scales=req.n_scales)
        image = np.array(req.image)
        # Identity operators: source-plane denoising (no lens model)
        def H(x): return x
        def HT(y): return y
        
        recon = st.solve_sparse_source_fista(
            image, H, HT, 
            thresholds=np.full(req.n_scales, req.lambda_reg), 
            max_iter=req.max_iter
        )
        recon_norm = recon / (recon.max() + 1e-9)
        
        import base64
        import io
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(2, 2))
        ax.imshow(recon_norm, cmap="magma", origin="lower")
        ax.axis("off")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=80, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        
        return {
            "max_intensity": float(recon.max()),
            "sparsity_fraction": float(np.mean(np.abs(recon) < 1e-4)),
            "image_b64": img_b64
        }
    except Exception as e:
        logger.exception("Starlet Error")
        raise HTTPException(status_code=500, detail=str(e))

class SEDRequest(BaseModel):
    flux_g: float
    flux_r: float
    flux_i: float

@router.post("/sed")
async def run_sed(req: SEDRequest):
    try:
        from src.ml.multiband_sed import SEDMorphologyJointLikelihood
        engine = SEDMorphologyJointLikelihood(num_bands=3)
        # Build per-band images scaled by the provided flux values
        fluxes = [req.flux_g, req.flux_r, req.flux_i]
        rng = np.random.default_rng(0)
        shared_morph = rng.random((10, 10))
        images = [flux * shared_morph + rng.random((10, 10)) * 0.01 for flux in fluxes]
        noises = [np.ones((10, 10)) for _ in range(3)]
        lens_ops = [lambda x: x for _ in range(3)]
        
        amps = engine.optimize_sed_amplitudes_linear(images, noises, lens_ops, shared_morph)
        
        return {
            "amplitudes": amps.tolist(),
            "computation_mode": "synthetic_morphology_demo",
            "note": "Morphology is a fixed synthetic pattern; fluxes modulate amplitude ratios only.",
        }
    except Exception as e:
        logger.exception("SED Error")
        raise HTTPException(status_code=500, detail=str(e))

class EnvLinkerRequest(BaseModel):
    n_galaxies: int = Field(100)
    z_lens: float = Field(0.5)
    z_source: float = Field(1.5)
    seed: int = Field(42, description="RNG seed for reproducibility")

@router.post("/env_linker")
async def run_env_linker(req: EnvLinkerRequest):
    try:
        from src.data.environmental_linker import EnvironmentalLinker
        import pandas as pd
        import tempfile
        import os

        rng = np.random.default_rng(req.seed)
        ra = rng.uniform(-0.1, 0.1, req.n_galaxies)
        dec = rng.uniform(-0.1, 0.1, req.n_galaxies)
        z = rng.uniform(0.1, 2.0, req.n_galaxies)
        mass = rng.uniform(1e10, 1e12, req.n_galaxies)
        df = pd.DataFrame({"ra": ra, "dec": dec, "redshift": z, "mass_msun": mass})
        
        fd, path = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        try:
            df.to_csv(path, index=False)
            
            linker = EnvironmentalLinker(main_lens_ra=0.0, main_lens_dec=0.0, main_lens_z=req.z_lens, source_z=req.z_source)
            kappa_ext = linker.compute_kappa_ext(path, format='csv')
        finally:
            if os.path.exists(path):
                os.remove(path)
        
        return {"kappa_ext": float(kappa_ext), "n_included": len(df[df['redshift'] <= req.z_source])}
    except Exception as e:
        logger.exception("EnvLinker Error")
        raise HTTPException(status_code=500, detail=str(e))

class ConsistencyRequest(BaseModel):
    M_vir: float = Field(..., description="Virial mass")
    r_s: float = Field(..., description="Scale radius")
    ellipticity: float = Field(..., description="Ellipticity")

@router.post("/consistency_gate")
async def run_consistency_gate(req: ConsistencyRequest):
    try:
        if req.M_vir <= 0:
            raise HTTPException(status_code=422, detail="M_vir must be positive")
        from scripts.scientific_consistency_gate import ScientificValidator
        val = ScientificValidator()
        
        # Stellar-to-halo mass ratio via Moster et al. (2010) SHMR approximation
        L_v = 10**(0.3 * (np.log10(req.M_vir) - 11.5) + 10.2)
        stellar_mass = L_v * 2.0
        dm_mass = req.M_vir - stellar_mass
        
        params = {
            "total_mass": req.M_vir,
            "total_luminosity": L_v,
            "dm_mass": dm_mass,
            "stellar_mass": stellar_mass
        }
        
        res = val.validate_inferred_model(params)
        is_valid = bool(res["passed"])
        
        M_L = float(req.M_vir / L_v)

        # Dark matter fraction proxy via Moster et al. (2010) SHMR
        log_Mstar_Mhalo = -1.405 + 0.325 * max(0, np.log10(req.M_vir) - 11.5)
        f_DM_proxy = 1.0 - 10**log_Mstar_Mhalo
        f_DM_proxy = float(np.clip(f_DM_proxy, 0.0, 1.0))
        
        return {
            "is_valid": is_valid,
            "M_L_ratio": M_L,
            "f_DM_proxy": f_DM_proxy,
            "computation_mode": "demo_approximation",
            "note": "M_L_ratio and f_DM_proxy use analytical approximations. Use full inference for publication.",
            "reason": "Within empirical 3-sigma bounds" if is_valid else "Astro-physical violation"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("ConsistencyGate Error")
        raise HTTPException(status_code=500, detail=str(e))

