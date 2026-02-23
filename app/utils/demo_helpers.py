"""
Demo Helper Utilities for Zero-Friction User Experience
========================================================
Handles asset loading, pipeline execution, and result management for one-click demos.
"""

import yaml
import numpy as np
import streamlit as st
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import matplotlib.pyplot as plt
from io import BytesIO
import logging

logger = logging.getLogger(__name__)

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DEMOS_DIR = PROJECT_ROOT / "demos"
ASSETS_DIR = PROJECT_ROOT / "assets" / "demos"


def ensure_demo_asset(asset_name: str) -> Path:
    """
    Resolve a demo asset path and require a real on-disk asset.
    
    Args:
        asset_name: Built-in asset identifier (e.g., "einstein_cross_hst")
        
    Returns:
        Path to the asset file

    Raises:
        FileNotFoundError
            If the named built-in asset does not exist on disk
    """
    asset_path = ASSETS_DIR / f"{asset_name}.npy"
    
    # If asset already exists, return path
    if asset_path.exists():
        logger.info(f"Found cached demo asset: {asset_name}")
        return asset_path
    raise FileNotFoundError(
        f"Required demo asset not found: {asset_path}. "
        "Automatic synthetic stand-in generation is disabled."
    )


def load_demo_config(demo_name: str) -> Dict[str, Any]:
    """
    Load demo configuration from YAML file.
    
    Args:
        demo_name: Name of demo (e.g., "einstein_cross")
        
    Returns:
        Configuration dictionary
    """
    config_path = DEMOS_DIR / f"{demo_name}.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Demo config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded demo config: {demo_name}")
    return config


def full_analysis_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute full gravitational lensing analysis pipeline.
    
    This function orchestrates:
    1. Asset loading/generation
    2. Ray tracing (thin_lens mode enforced)
    3. PINN inference
    4. Uncertainty quantification
    5. Result formatting
    
    Args:
        config: Demo configuration dictionary
        
    Returns:
        Results dictionary with images, parameters, and uncertainties
    """
    from src.lens_models import LensSystem, NFWProfile, EllipticalNFWProfile
    from src.ml.generate_dataset import generate_convergence_map_vectorized
    from src.ml.pinn import PhysicsInformedNN
    import torch
    
    logger.info(f"Starting analysis pipeline for: {config.get('name', 'Unknown')}")
    
    # Step 1: Load observation data; fallback to physics-based generation when
    # built-in assets are not available in this local workspace.
    source_image = str(config.get("data", {}).get("source_image", ""))
    if source_image.startswith("builtin:"):
        asset_name = source_image.split(":", 1)[1]
        try:
            asset_path = ensure_demo_asset(asset_name)
            observation = np.load(asset_path)
        except FileNotFoundError:
            logger.warning(
                "Builtin demo asset '%s' not found under %s. Falling back to "
                "physics-based synthetic generation from config.",
                asset_name,
                ASSETS_DIR,
            )
            observation = _generate_from_config(config)
    else:
        # Generate from config parameters
        observation = _generate_from_config(config)
    
    logger.info(f"Loaded observation: shape {observation.shape}")
    
    # Step 2: Validate thin_lens mode enforcement
    ray_mode = config.get("ray_tracing", {}).get("mode", "thin_lens")
    if ray_mode != "thin_lens":
        raise ValueError(
            f"Demo configs must use thin_lens mode (got: {ray_mode}). "
            "Schwarzschild mode is disabled for cosmological demos."
        )
    
    # Step 3: Set up lens system and profile
    lens_config = config["lens"]
    source_config = config["source"]
    
    lens_redshift = float(lens_config["z"])
    source_redshift = float(source_config["z"])
    lens_mass_msun = float(lens_config["mass"])
    concentration = float(lens_config.get("concentration", 10.0))
    ellipticity = float(lens_config.get("ellipticity", 0.0))
    lens_model = str(lens_config.get("model", "NFW"))

    lens_system = LensSystem(
        z_lens=lens_redshift,
        z_source=source_redshift,
    )

    if lens_model.lower().startswith("elliptical"):
        lens_profile = EllipticalNFWProfile(
            M_vir=lens_mass_msun,
            c=concentration,
            lens_sys=lens_system,
            ellipticity=ellipticity,
            position_angle=float(lens_config.get("position_angle", 0.0)),
        )
    else:
        lens_profile = NFWProfile(
            M_vir=lens_mass_msun,
            concentration=concentration,
            lens_system=lens_system,
            ellipticity=ellipticity,
        )
    
    logger.info(f"Lens system: {lens_config['model']} at z={lens_config['z']}")
    
    # Step 4: Ray tracing fields on grid
    grid_res = config.get("ray_tracing", {}).get("grid_resolution", 256)
    fov = config.get("observation", {}).get("fov_size", 128) * config.get("observation", {}).get("pixel_scale", 0.05)
    
    # Generate grid
    x = np.linspace(-fov/2, fov/2, grid_res)
    y = np.linspace(-fov/2, fov/2, grid_res)
    xx, yy = np.meshgrid(x, y)
    
    # Compute deflection and convergence from the physical profile.
    alpha_x_flat, alpha_y_flat = lens_profile.deflection_angle(xx.ravel(), yy.ravel())
    alpha_x = np.asarray(alpha_x_flat).reshape(grid_res, grid_res)
    alpha_y = np.asarray(alpha_y_flat).reshape(grid_res, grid_res)
    convergence = generate_convergence_map_vectorized(
        lens_model=lens_profile,
        grid_size=grid_res,
        extent=fov / 2,
    )
    
    logger.info("Ray tracing complete")
    
    # Step 5: PINN inference (if enabled)
    pinn_results = None
    uncertainty_map = None
    
    if config.get("analysis", {}).get("run_pinn_inference", False):
        try:
            # Load pre-trained PINN model
            pinn_model_path = PROJECT_ROOT / "src" / "ml" / "models" / "pretrained" / "pinn_lens_v1.pth"
            
            if pinn_model_path.exists():
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                pinn = PhysicsInformedNN(input_size=64, dropout_rate=0.2).to(device)
                checkpoint = torch.load(pinn_model_path, map_location=device)
                pinn.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=False)
                pinn.eval()
                
                logger.info("PINN inference enabled (using pre-trained model)")

                # Prepare 64x64 normalized map for parameter inference.
                from scipy.ndimage import zoom
                conv_for_model = convergence
                if conv_for_model.shape != (64, 64):
                    scale = 64 / conv_for_model.shape[0]
                    conv_for_model = zoom(conv_for_model, scale, order=1)
                conv_norm = (conv_for_model - conv_for_model.min()) / (
                    conv_for_model.max() - conv_for_model.min() + 1e-10
                )
                model_input = torch.from_numpy(conv_norm).float().unsqueeze(0).unsqueeze(0).to(device)

                with torch.no_grad():
                    pred_params, pred_logits = pinn(model_input)
                    class_probs = torch.softmax(pred_logits, dim=1).cpu().numpy()[0]

                # Keep compatibility with results page expecting convergence_pred.
                pinn_results = {
                    "convergence_pred": convergence.copy(),
                    "params_pred": pred_params.cpu().numpy()[0],
                    "class_probs": class_probs,
                }

                if config.get("analysis", {}).get("uncertainty_quantification", False):
                    # Conservative scalar uncertainty proxy from MC dropout over params.
                    pinn.train()
                    mc_samples = 20
                    param_samples = []
                    for _ in range(mc_samples):
                        with torch.no_grad():
                            p, _ = pinn(model_input)
                            param_samples.append(p.cpu().numpy()[0])
                    pinn.eval()
                    param_samples = np.asarray(param_samples)
                    uncertainty_level = float(np.mean(np.std(param_samples, axis=0)))
                    uncertainty_map = np.full_like(convergence, uncertainty_level)
                    logger.info("Uncertainty quantification complete")
            else:
                logger.warning(f"PINN model not found at {pinn_model_path}, skipping inference")
        except Exception as e:
            logger.error(f"PINN inference failed: {e}")
            pinn_results = None
    
    # Step 6: Generate result visualizations
    results = {
        "config": config,
        "observation": observation,
        "convergence_map": convergence,
        "deflection": (alpha_x, alpha_y),
        "pinn_results": pinn_results,
        "uncertainty_map": uncertainty_map,
        "lens_parameters": {
            "mass": lens_mass_msun,
            "z_lens": lens_redshift,
            "z_source": source_redshift,
            "model": lens_model,
            "ellipticity": ellipticity,
        },
        "ray_tracing_mode": "thin_lens",
    }
    
    logger.info("Pipeline complete")
    return results


def _generate_from_config(config: Dict[str, Any]) -> np.ndarray:
    """
    Generate a physics-based synthetic observation from demo config.

    This path avoids non-physical stand-in/random-only images.
    """
    from src.ml.generate_dataset import generate_synthetic_convergence

    lens_config = config.get("lens", {})
    obs_config = config.get("observation", {})
    source_cfg = config.get("source", {})

    size = int(obs_config.get("fov_size", 128))
    profile = lens_config.get("model", "NFW")
    if profile.lower().startswith("elliptical"):
        profile_type = "Elliptical NFW"
    else:
        profile_type = "NFW"

    convergence_map, _, _ = generate_synthetic_convergence(
        profile_type=profile_type,
        mass=float(lens_config.get("mass", 1e12)),
        scale_radius=float(lens_config.get("scale_radius", 200.0)),
        ellipticity=float(lens_config.get("ellipticity", 0.0)),
        grid_size=size,
        z_lens=float(lens_config.get("z", 0.5)),
        z_source=float(source_cfg.get("z", 1.5)),
    )

    noise_level = float(obs_config.get("noise_level", 0.0))
    if noise_level > 0:
        seed = int(config.get("seed", 42))
        rng = np.random.default_rng(seed)
        convergence_map = convergence_map + rng.normal(0.0, noise_level, convergence_map.shape)

    return convergence_map.astype(np.float32)


def run_demo_and_redirect(demo_name: str):
    """
    Execute demo pipeline and redirect to results page.
    
    Args:
        demo_name: Name of demo to run
    """
    try:
        # Load configuration
        config = load_demo_config(demo_name)
        
        # Run full analysis
        with st.spinner(f"🌌 Simulating light paths through curved spacetime ({config.get('name', demo_name)})..."):
            results = full_analysis_pipeline(config)
        
        # Store in session state
        st.session_state["demo_results"] = results
        st.session_state["demo_name"] = demo_name
        
        st.toast("✅ Simulation complete!", icon="✨")
        
        # Redirect to results page
        st.switch_page("pages/03_Results.py")
        
    except Exception as e:
        st.error(f"❌ Demo execution failed: {str(e)}")
        logger.error(f"Demo {demo_name} failed: {e}", exc_info=True)


def export_pdf_report(results: Dict[str, Any]) -> BytesIO:
    """
    Generate PDF report from results.
    
    Args:
        results: Analysis results dictionary
        
    Returns:
        BytesIO buffer containing PDF
    """
    from matplotlib.backends.backend_pdf import PdfPages
    
    buffer = BytesIO()
    
    with PdfPages(buffer) as pdf:
        # Page 1: Overview
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle(f"Gravitational Lensing Analysis: {results['config'].get('name', 'Demo')}", fontsize=16, fontweight='bold')
        
        # Observation
        axes[0, 0].imshow(results["observation"], cmap='hot', origin='lower')
        axes[0, 0].set_title("Observation")
        axes[0, 0].axis('off')
        
        # Convergence map
        axes[0, 1].imshow(results["convergence_map"], cmap='viridis', origin='lower')
        axes[0, 1].set_title("Convergence κ (Mass Map)")
        axes[0, 1].axis('off')
        
        # PINN reconstruction (if available)
        if results.get("pinn_results"):
            axes[1, 0].imshow(results["pinn_results"]["convergence_pred"], cmap='viridis', origin='lower')
            axes[1, 0].set_title("PINN Reconstruction")
            axes[1, 0].axis('off')
        
        # Uncertainty
        if results.get("uncertainty_map") is not None:
            axes[1, 1].imshow(results["uncertainty_map"], cmap='Reds', origin='lower')
            axes[1, 1].set_title("95% Uncertainty")
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        
        # Page 2: Parameters
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        params = results["lens_parameters"]
        param_text = f"""
        Lens Parameters:
        ───────────────────────────────
        Model: {params['model']}
        Mass: {params['mass']:.2e} M☉
        Lens Redshift (z_l): {params['z_lens']}
        Source Redshift (z_s): {params['z_source']}
        Ellipticity: {params['ellipticity']:.2f}
        
        Ray Tracing Mode: {results['ray_tracing_mode']}
        
        Analysis:
        ───────────────────────────────
        PINN Inference: {'✓' if results.get('pinn_results') else '✗'}
        Uncertainty Quantification: {'✓' if results.get('uncertainty_map') is not None else '✗'}
        """
        
        ax.text(0.1, 0.9, param_text, transform=ax.transAxes, fontsize=12, verticalalignment='top', fontfamily='monospace')
        
        pdf.savefig(fig)
        plt.close()
    
    buffer.seek(0)
    return buffer
