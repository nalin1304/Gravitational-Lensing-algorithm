#!/usr/bin/env python3
"""Checkpoint-backed uncertainty calibration for synthetic NFW lens analogs.

This script trains or loads a Bayesian surrogate that maps
``(x, y, r, x_norm, y_norm, r_norm, log10(M_vir), c, z_l, z_s) -> kappa(x, y)`` and evaluates its
Monte-Carlo-dropout calibration on held-out synthetic systems.

The resulting calibration artifact is publication-valid for the synthetic
NFW-analog regime represented by the training prior. It is not an observational
SLACS posterior calibration and must not be described as such.

References
----------
Gal, Y. & Ghahramani, Z. (2016), PMLR 48, 1050.
Guo, C. et al. (2017), PMLR 70, 1321.
Bartelmann, M. (1996), A&A 313, 697.
Wright, C. O. & Brainerd, T. G. (2000), ApJ 534, 34.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import norm
from torch.utils.data import DataLoader, TensorDataset

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

mpl_cache_dir = Path(tempfile.gettempdir()) / "gravitational_lensing_matplotlib"
mpl_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache_dir))

from src.lens_models.lens_system import LensSystem
from src.lens_models.mass_profiles import NFWProfile
from src.ml.generate_dataset import generate_convergence_map_vectorized
from src.ml.uncertainty.bayesian_uq import BayesianPINN
from src.utils.common import set_random_seed

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


CONFIDENCE_LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
DEFAULT_CHECKPOINT_PATH = Path("models/bayesian_uq_synthetic.pt")
DEFAULT_DROPOUT_RATE = 0.04
DEFAULT_HIDDEN_DIMS = [128, 128, 64]
SYNTHETIC_SIGMA_V_RANGE = (160.0, 320.0)
SYNTHETIC_Z_L_RANGE = (0.10, 0.40)
SYNTHETIC_ZS_OFFSET_RANGE = (0.25, 1.20)
SYNTHETIC_CONCENTRATION_RANGE = (5.0, 12.0)
MIN_UNCERTAINTY_FLOOR = 1.0e-6
DEFAULT_CALIBRATION_SHRINKAGE = 0.5


@dataclass(frozen=True)
class LensAnalogSpec:
    """Parameters for a galaxy-scale synthetic NFW analog system."""

    name: str
    z_l: float
    z_s: float
    sigma_v_kms: float
    concentration: float

    @property
    def virial_mass_msun(self) -> float:
        """Approximate virial mass from Faber-Jackson-style sigma scaling."""
        return float(1.0e12 * (self.sigma_v_kms / 200.0) ** 4)

    @property
    def extent_arcsec(self) -> float:
        """Use a deterministic field of view tied to the point-mass scale."""
        lens_system = LensSystem(z_lens=self.z_l, z_source=self.z_s)
        theta_e_arcsec = lens_system.einstein_radius_scale(self.virial_mass_msun)
        return float(max(2.0, 2.2 * theta_e_arcsec))


@dataclass(frozen=True)
class TrainingConfig:
    """Training hyperparameters for the Bayesian surrogate."""

    train_systems: int
    val_systems: int
    train_grid: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    dropout_rate: float
    n_samples: int
    seed: int


@dataclass
class ModelBundle:
    """Trained model and normalization/calibration metadata."""

    model: BayesianPINN
    feature_mean: np.ndarray
    feature_std: np.ndarray
    target_mean: float
    target_std: float
    uncertainty_scale: float
    config: TrainingConfig
    checkpoint_path: Path
    calibration_ece: float
    calibration_coverage_90: float
    calibrated_confidence_levels: np.ndarray
    calibrated_z_scores: np.ndarray
    calibration_shrinkage: float


def _slacs_analogs() -> list[LensAnalogSpec]:
    return [
        LensAnalogSpec("SDSS J0946+1006", z_l=0.222, z_s=0.609, sigma_v_kms=263.0, concentration=8.4),
        LensAnalogSpec("SDSS J1250+0523", z_l=0.232, z_s=0.795, sigma_v_kms=252.0, concentration=8.2),
        LensAnalogSpec("SDSS J1402+6321", z_l=0.205, z_s=0.481, sigma_v_kms=267.0, concentration=8.5),
        LensAnalogSpec("SDSS J0252+0039", z_l=0.280, z_s=0.982, sigma_v_kms=164.0, concentration=9.0),
        LensAnalogSpec("SDSS J0037-0942", z_l=0.195, z_s=0.632, sigma_v_kms=279.0, concentration=8.0),
    ]


def _random_specs(count: int, rng: np.random.Generator, prefix: str) -> list[LensAnalogSpec]:
    specs: list[LensAnalogSpec] = []
    for index in range(count):
        z_l = float(rng.uniform(*SYNTHETIC_Z_L_RANGE))
        z_s = float(z_l + rng.uniform(*SYNTHETIC_ZS_OFFSET_RANGE))
        sigma_v = float(rng.uniform(*SYNTHETIC_SIGMA_V_RANGE))
        concentration = float(rng.uniform(*SYNTHETIC_CONCENTRATION_RANGE))
        specs.append(
            LensAnalogSpec(
                name=f"{prefix}_{index:03d}",
                z_l=z_l,
                z_s=z_s,
                sigma_v_kms=sigma_v,
                concentration=concentration,
            )
        )
    return specs


def _build_system_arrays(spec: LensAnalogSpec, grid_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return feature matrix, target kappa map, and raw sample map for one system."""
    lens_system = LensSystem(z_lens=spec.z_l, z_source=spec.z_s)
    lens_model = NFWProfile(
        M_vir=spec.virial_mass_msun,
        concentration=spec.concentration,
        lens_system=lens_system,
    )
    extent_arcsec = spec.extent_arcsec
    coordinates = np.linspace(-extent_arcsec, extent_arcsec, grid_size, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(coordinates, coordinates)
    convergence_map = generate_convergence_map_vectorized(
        lens_model=lens_model,
        grid_size=grid_size,
        extent=extent_arcsec,
    ).astype(np.float32)
    feature_matrix = np.column_stack(
        [
            x_grid.ravel(),
            y_grid.ravel(),
            np.sqrt(x_grid**2 + y_grid**2).ravel(),
            (x_grid / extent_arcsec).ravel(),
            (y_grid / extent_arcsec).ravel(),
            (np.sqrt(x_grid**2 + y_grid**2) / extent_arcsec).ravel(),
            np.full(x_grid.size, np.log10(spec.virial_mass_msun), dtype=np.float32),
            np.full(x_grid.size, spec.concentration, dtype=np.float32),
            np.full(x_grid.size, spec.z_l, dtype=np.float32),
            np.full(x_grid.size, spec.z_s, dtype=np.float32),
        ]
    ).astype(np.float32)
    return feature_matrix, np.log1p(convergence_map.ravel()).astype(np.float32), convergence_map


def _stack_dataset(specs: Iterable[LensAnalogSpec], grid_size: int) -> tuple[np.ndarray, np.ndarray]:
    feature_blocks: list[np.ndarray] = []
    target_blocks: list[np.ndarray] = []
    for spec in specs:
        features, targets, _ = _build_system_arrays(spec, grid_size)
        feature_blocks.append(features)
        target_blocks.append(targets[:, None])
    return np.vstack(feature_blocks), np.vstack(target_blocks)


def _normalize_features(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (features - mean) / std


def _create_dataloaders(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    val_features: np.ndarray,
    val_targets: np.ndarray,
    batch_size: int,
) -> tuple[DataLoader[tuple[torch.Tensor, torch.Tensor]], DataLoader[tuple[torch.Tensor, torch.Tensor]]]:
    train_dataset = TensorDataset(
        torch.from_numpy(train_features).float(),
        torch.from_numpy(train_targets).float(),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(val_features).float(),
        torch.from_numpy(val_targets).float(),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def _train_model(
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    config: TrainingConfig,
    device: torch.device,
) -> tuple[BayesianPINN, float]:
    model = BayesianPINN(
        input_dim=10,
        output_dim=1,
        hidden_dims=DEFAULT_HIDDEN_DIMS,
        dropout_rate=config.dropout_rate,
        activation="tanh",
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(config.epochs, 1))
    best_state = None
    best_val_loss = float("inf")

    for epoch in range(config.epochs):
        model.train()
        train_losses: list[float] = []
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(batch_features)
            loss = F.mse_loss(predictions, batch_targets)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))
        scheduler.step()

        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)
                predictions = model(batch_features)
                val_losses.append(float(F.mse_loss(predictions, batch_targets).cpu().item()))
        mean_val_loss = float(np.mean(val_losses))
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}

        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == config.epochs:
            mean_train_loss = float(np.mean(train_losses))
            print(
                f"  epoch {epoch + 1:02d}/{config.epochs}: "
                f"train_loss={mean_train_loss:.6f} val_loss={mean_val_loss:.6f}"
            )

    if best_state is None:
        raise RuntimeError("Training finished without producing a valid state dict.")
    model.load_state_dict(best_state)
    model.eval()
    return model, best_val_loss


def _predict_samples(
    model: BayesianPINN,
    spec: LensAnalogSpec,
    grid_size: int,
    n_samples: int,
    device: torch.device,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    target_mean: float,
    target_std: float,
) -> tuple[np.ndarray, np.ndarray]:
    features, _, ground_truth = _build_system_arrays(spec, grid_size)
    normalized_features = _normalize_features(features, feature_mean, feature_std)
    input_tensor = torch.from_numpy(normalized_features).float().to(device)

    mean_pred, _, sample_predictions = model.predict_with_uncertainty(
        input_tensor,
        n_samples=n_samples,
        return_samples=True,
    )
    del mean_pred
    sample_predictions_np = sample_predictions.detach().cpu().numpy()[:, :, 0]
    sample_predictions_np = sample_predictions_np * target_std + target_mean
    kappa_samples = np.expm1(sample_predictions_np)
    kappa_samples = np.maximum(kappa_samples, 0.0).reshape(n_samples, grid_size, grid_size)
    return ground_truth, kappa_samples


def _rescale_samples(kappa_samples: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean_prediction = np.mean(kappa_samples, axis=0)
    scaled_samples = mean_prediction[None, :, :] + scale * (kappa_samples - mean_prediction[None, :, :])
    scaled_samples = np.maximum(scaled_samples, 0.0)
    scaled_mean = np.mean(scaled_samples, axis=0)
    scaled_std = np.maximum(np.std(scaled_samples, axis=0), MIN_UNCERTAINTY_FLOOR)
    return scaled_samples, scaled_mean, scaled_std


def _compute_coverage_from_samples(
    ground_truth: np.ndarray,
    sample_predictions: np.ndarray,
    confidence_levels: np.ndarray = CONFIDENCE_LEVELS,
    calibrated_z_scores: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    coverages: list[float] = []
    mean_prediction = np.mean(sample_predictions, axis=0)
    std_prediction = np.maximum(np.std(sample_predictions, axis=0), MIN_UNCERTAINTY_FLOOR)
    if calibrated_z_scores is None:
        calibrated_z_scores = norm.ppf((1.0 + confidence_levels) / 2.0)
    for z_score in calibrated_z_scores:
        lower = mean_prediction - float(z_score) * std_prediction
        upper = mean_prediction + float(z_score) * std_prediction
        within = np.mean((ground_truth >= lower) & (ground_truth <= upper))
        coverages.append(float(within))
    return confidence_levels, np.asarray(coverages, dtype=np.float64)


def _compute_ece(confidence_levels: np.ndarray, empirical_coverages: np.ndarray) -> float:
    return float(np.mean(np.abs(empirical_coverages - confidence_levels)))


def _derive_calibrated_z_scores(
    cached_predictions: list[tuple[np.ndarray, np.ndarray]],
    scale: float,
    confidence_levels: np.ndarray = CONFIDENCE_LEVELS,
    shrinkage: float = DEFAULT_CALIBRATION_SHRINKAGE,
) -> np.ndarray:
    normalized_residuals: list[np.ndarray] = []
    for ground_truth, samples in cached_predictions:
        _, mean_prediction, std_prediction = _rescale_samples(samples, scale)
        residual_ratio = np.abs(ground_truth - mean_prediction) / std_prediction
        normalized_residuals.append(residual_ratio.ravel())

    stacked_ratios = np.concatenate(normalized_residuals)
    empirical_z_scores = []
    for confidence in confidence_levels:
        empirical_z_scores.append(float(np.quantile(stacked_ratios, confidence)))
    empirical = np.asarray(empirical_z_scores, dtype=float)
    gaussian = norm.ppf((1.0 + confidence_levels) / 2.0)
    clipped_shrinkage = float(np.clip(shrinkage, 0.0, 1.0))
    return clipped_shrinkage * empirical + (1.0 - clipped_shrinkage) * gaussian


def _calibration_objective(
    model: BayesianPINN,
    validation_specs: list[LensAnalogSpec],
    grid_size: int,
    n_samples: int,
    device: torch.device,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    target_mean: float,
    target_std: float,
) -> tuple[float, float, float, np.ndarray]:
    # The optimum can sit below 0.6 for over-dispersed checkpoints, so keep the
    # search wide enough to avoid boundary-clipping the calibration solution.
    candidate_scales = np.linspace(0.2, 2.4, 23)
    best_scale = 1.0
    best_ece = float("inf")
    best_coverage_90 = float("nan")
    best_z_scores = norm.ppf((1.0 + CONFIDENCE_LEVELS) / 2.0)

    cached_predictions: list[tuple[np.ndarray, np.ndarray]] = []
    for spec in validation_specs:
        cached_predictions.append(
            _predict_samples(
                model=model,
                spec=spec,
                grid_size=grid_size,
                n_samples=n_samples,
                device=device,
                feature_mean=feature_mean,
                feature_std=feature_std,
                target_mean=target_mean,
                target_std=target_std,
            )
        )

    for scale in candidate_scales:
        calibrated_z_scores = _derive_calibrated_z_scores(
            cached_predictions,
            float(scale),
            shrinkage=DEFAULT_CALIBRATION_SHRINKAGE,
        )
        eces: list[float] = []
        coverage_90_values: list[float] = []
        for ground_truth, samples in cached_predictions:
            scaled_samples, _, _ = _rescale_samples(samples, float(scale))
            confidence_levels, coverages = _compute_coverage_from_samples(
                ground_truth,
                scaled_samples,
                confidence_levels=CONFIDENCE_LEVELS,
                calibrated_z_scores=calibrated_z_scores,
            )
            eces.append(_compute_ece(confidence_levels, coverages))
            coverage_90_values.append(float(coverages[np.where(np.isclose(confidence_levels, 0.9))[0][0]]))
        mean_ece = float(np.mean(eces))
        if mean_ece < best_ece:
            best_scale = float(scale)
            best_ece = mean_ece
            best_coverage_90 = float(np.mean(coverage_90_values))
            best_z_scores = calibrated_z_scores

    return best_scale, best_ece, best_coverage_90, best_z_scores


def _save_checkpoint(bundle: ModelBundle) -> None:
    bundle.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": bundle.model.state_dict(),
        "metadata": {
            "feature_mean": bundle.feature_mean.tolist(),
            "feature_std": bundle.feature_std.tolist(),
            "target_mean": bundle.target_mean,
            "target_std": bundle.target_std,
            "uncertainty_scale": bundle.uncertainty_scale,
            "calibration_ece": bundle.calibration_ece,
            "calibration_coverage_90": bundle.calibration_coverage_90,
            "calibrated_confidence_levels": bundle.calibrated_confidence_levels.tolist(),
            "calibrated_z_scores": bundle.calibrated_z_scores.tolist(),
            "calibration_shrinkage": bundle.calibration_shrinkage,
            "config": asdict(bundle.config),
            "input_features": [
                "x_arcsec",
                "y_arcsec",
                "radius_arcsec",
                "x_over_extent",
                "y_over_extent",
                "radius_over_extent",
                "log10_mvir",
                "concentration",
                "z_l",
                "z_s",
            ],
            "input_dim": int(bundle.feature_mean.shape[0]),
            "target_transform": "log1p(kappa)",
            "evaluation_scope": "synthetic_nfw_analogs_only",
            "dropout_rate": bundle.config.dropout_rate,
        },
    }
    torch.save(payload, bundle.checkpoint_path)


def _load_checkpoint(path: Path, device: torch.device) -> ModelBundle:
    checkpoint = torch.load(path, map_location=device)
    metadata = checkpoint["metadata"]
    config = TrainingConfig(**metadata["config"])
    input_dim = int(metadata.get("input_dim", len(metadata["feature_mean"])))
    model = BayesianPINN(
        input_dim=input_dim,
        output_dim=1,
        hidden_dims=DEFAULT_HIDDEN_DIMS,
        dropout_rate=float(metadata.get("dropout_rate", config.dropout_rate)),
        activation="tanh",
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return ModelBundle(
        model=model,
        feature_mean=np.asarray(metadata["feature_mean"], dtype=np.float32),
        feature_std=np.asarray(metadata["feature_std"], dtype=np.float32),
        target_mean=float(metadata["target_mean"]),
        target_std=float(metadata["target_std"]),
        uncertainty_scale=float(metadata["uncertainty_scale"]),
        config=config,
        checkpoint_path=path,
        calibration_ece=float(metadata.get("calibration_ece", float("nan"))),
        calibration_coverage_90=float(metadata.get("calibration_coverage_90", float("nan"))),
        calibrated_confidence_levels=np.asarray(
            metadata.get("calibrated_confidence_levels", CONFIDENCE_LEVELS.tolist()),
            dtype=np.float32,
        ),
        calibrated_z_scores=np.asarray(
            metadata.get(
                "calibrated_z_scores",
                norm.ppf((1.0 + CONFIDENCE_LEVELS) / 2.0).tolist(),
            ),
            dtype=np.float32,
        ),
        calibration_shrinkage=float(metadata.get("calibration_shrinkage", DEFAULT_CALIBRATION_SHRINKAGE)),
    )


def _train_or_load_bundle(args: argparse.Namespace, device: torch.device) -> ModelBundle:
    checkpoint_path = Path(args.model)
    if checkpoint_path.exists() and not args.force_retrain:
        print(f"Loading checkpoint-backed Bayesian UQ model: {checkpoint_path}")
        return _load_checkpoint(checkpoint_path, device)

    print("Training checkpoint-backed Bayesian UQ surrogate...")
    config = TrainingConfig(
        train_systems=args.train_systems,
        val_systems=args.val_systems,
        train_grid=args.train_grid,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout_rate=args.dropout_rate,
        n_samples=args.n_samples,
        seed=args.seed,
    )
    rng = np.random.default_rng(args.seed)
    train_specs = _random_specs(config.train_systems, rng, prefix="train")
    val_specs = _random_specs(config.val_systems, rng, prefix="val")

    train_features_raw, train_targets_raw = _stack_dataset(train_specs, grid_size=config.train_grid)
    val_features_raw, val_targets_raw = _stack_dataset(val_specs, grid_size=config.train_grid)

    feature_mean = train_features_raw.mean(axis=0)
    feature_std = train_features_raw.std(axis=0) + 1.0e-6
    target_mean = float(train_targets_raw.mean())
    target_std = float(train_targets_raw.std() + 1.0e-6)

    train_features = _normalize_features(train_features_raw, feature_mean, feature_std)
    val_features = _normalize_features(val_features_raw, feature_mean, feature_std)
    train_targets = (train_targets_raw - target_mean) / target_std
    val_targets = (val_targets_raw - target_mean) / target_std

    train_loader, val_loader = _create_dataloaders(
        train_features=train_features,
        train_targets=train_targets,
        val_features=val_features,
        val_targets=val_targets,
        batch_size=config.batch_size,
    )
    model, best_val_loss = _train_model(train_loader, val_loader, config=config, device=device)
    print(f"Best validation loss: {best_val_loss:.6f}")

    uncertainty_scale, calibration_ece, calibration_coverage_90, calibrated_z_scores = _calibration_objective(
        model=model,
        validation_specs=val_specs,
        grid_size=args.grid,
        n_samples=args.n_samples,
        device=device,
        feature_mean=feature_mean,
        feature_std=feature_std,
        target_mean=target_mean,
        target_std=target_std,
    )
    print(
        "Validation-set uncertainty scaling: "
        f"scale={uncertainty_scale:.3f} mean_ece={calibration_ece:.4f} "
        f"coverage_90={calibration_coverage_90:.3f}"
    )

    bundle = ModelBundle(
        model=model,
        feature_mean=feature_mean.astype(np.float32),
        feature_std=feature_std.astype(np.float32),
        target_mean=target_mean,
        target_std=target_std,
        uncertainty_scale=uncertainty_scale,
        config=config,
        checkpoint_path=checkpoint_path,
        calibration_ece=calibration_ece,
        calibration_coverage_90=calibration_coverage_90,
        calibrated_confidence_levels=CONFIDENCE_LEVELS.copy(),
        calibrated_z_scores=calibrated_z_scores.astype(np.float32),
        calibration_shrinkage=DEFAULT_CALIBRATION_SHRINKAGE,
    )
    _save_checkpoint(bundle)
    print(f"Saved checkpoint-backed Bayesian UQ model: {checkpoint_path}")
    return bundle


def _generate_calibration_figure(all_results: list[dict[str, object]], outdir: Path) -> None:
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available; skipping uncertainty calibration figure.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    reliability_axis = axes[0]
    reliability_axis.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration", alpha=0.7)
    for result in all_results:
        reliability_axis.plot(
            result["confidence_levels"],
            result["coverages"],
            "o-",
            ms=3,
            lw=1.0,
            alpha=0.45,
        )
    mean_coverages = np.mean([result["coverages"] for result in all_results], axis=0)
    reliability_axis.plot(
        all_results[0]["confidence_levels"],
        mean_coverages,
        "s-",
        color="#00d4ff",
        markersize=5,
        lw=2,
        label="Mean held-out coverage",
    )
    reliability_axis.set_xlabel("Expected coverage")
    reliability_axis.set_ylabel("Empirical coverage")
    reliability_axis.set_title("(a) Reliability diagram")
    reliability_axis.set_xlim(0, 1)
    reliability_axis.set_ylim(0, 1)
    reliability_axis.grid(True, alpha=0.3)
    reliability_axis.legend(fontsize=8)

    ece_axis = axes[1]
    names = [str(result["name"]).replace("SDSS ", "") for result in all_results]
    eces = [float(result["ece"]) for result in all_results]
    colors = ["#2ecc71" if ece < 0.05 else "#f1c40f" if ece < 0.10 else "#e74c3c" for ece in eces]
    ece_axis.barh(names, eces, color=colors, edgecolor="#1f2937", height=0.6)
    ece_axis.axvline(0.05, color="#94a3b8", linestyle="--", lw=1.2, alpha=0.8, label="ECE = 0.05")
    ece_axis.set_xlabel("Expected calibration error")
    ece_axis.set_title("(b) Per-system ECE")
    ece_axis.grid(True, alpha=0.3, axis="x")
    ece_axis.legend(fontsize=8)

    scatter_axis = axes[2]
    for result in all_results:
        scatter_axis.scatter(
            result["pixel_uncertainties"],
            result["pixel_errors"],
            alpha=0.02,
            s=2,
            c="#38bdf8",
        )
    max_val = max(
        max(float(np.max(result["pixel_uncertainties"])) for result in all_results),
        max(float(np.max(result["pixel_errors"])) for result in all_results),
    )
    scatter_axis.plot([0, max_val], [0, max_val], "--", color="#f97316", lw=1.5, label=r"$\sigma = |error|$")
    scatter_axis.set_xlabel("Predicted uncertainty")
    scatter_axis.set_ylabel("Absolute error")
    scatter_axis.set_title("(c) Uncertainty-error alignment")
    scatter_axis.set_xlim(0, max_val * 1.05)
    scatter_axis.set_ylim(0, max_val * 1.05)
    scatter_axis.grid(True, alpha=0.3)
    scatter_axis.legend(fontsize=8)

    plt.tight_layout()
    output_path = outdir / "uncertainty_calibration.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Calibration figure saved: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, default=64, help="Evaluation grid size.")
    parser.add_argument("--train-grid", type=int, default=24, help="Training grid size for synthetic systems.")
    parser.add_argument("--n-samples", type=int, default=30, help="MC Dropout forward passes at evaluation time.")
    parser.add_argument("--outdir", type=str, default="results", help="Output directory for calibration artifacts.")
    parser.add_argument("--model", type=str, default=str(DEFAULT_CHECKPOINT_PATH), help="Checkpoint path to load/save.")
    parser.add_argument("--force-retrain", action="store_true", help="Retrain even when a checkpoint already exists.")
    parser.add_argument("--train-systems", type=int, default=32, help="Number of random training systems.")
    parser.add_argument("--val-systems", type=int, default=10, help="Number of random validation systems.")
    parser.add_argument("--epochs", type=int, default=18, help="Training epochs when retraining is required.")
    parser.add_argument("--batch-size", type=int, default=1024, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=1.0e-3, help="Adam learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1.0e-6, help="Adam weight decay.")
    parser.add_argument("--dropout-rate", type=float, default=DEFAULT_DROPOUT_RATE, help="MC Dropout probability.")
    parser.add_argument("--seed", type=int, default=21, help="Global random seed.")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    set_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 79)
    print("  UNCERTAINTY CALIBRATION ANALYSIS — CHECKPOINT-BACKED MC DROPOUT")
    print("=" * 79)
    print(f"  Device: {device.type} | eval grid: {args.grid} | T={args.n_samples}")
    print(f"  Checkpoint: {args.model}")
    print("  Scope: synthetic held-out NFW analogs only (not observational posterior UQ)")
    print("=" * 79)

    bundle = _train_or_load_bundle(args, device)
    analogs = _slacs_analogs()
    all_results: list[dict[str, object]] = []

    for analog in analogs:
        print(f"\n▶ {analog.name}")
        ground_truth, raw_samples = _predict_samples(
            model=bundle.model,
            spec=analog,
            grid_size=args.grid,
            n_samples=args.n_samples,
            device=device,
            feature_mean=bundle.feature_mean,
            feature_std=bundle.feature_std,
            target_mean=bundle.target_mean,
            target_std=bundle.target_std,
        )
        scaled_samples, mean_prediction, std_prediction = _rescale_samples(raw_samples, bundle.uncertainty_scale)
        confidence_levels, coverages = _compute_coverage_from_samples(
            ground_truth,
            scaled_samples,
            confidence_levels=bundle.calibrated_confidence_levels,
            calibrated_z_scores=bundle.calibrated_z_scores,
        )
        pixel_errors = np.abs(ground_truth.ravel() - mean_prediction.ravel())
        pixel_uncertainties = std_prediction.ravel()
        ece = _compute_ece(confidence_levels, coverages)
        coverage_90 = float(coverages[np.where(np.isclose(confidence_levels, 0.9))[0][0]])
        correlation = float(np.corrcoef(pixel_uncertainties, pixel_errors)[0, 1])
        rmse = float(np.sqrt(np.mean((ground_truth - mean_prediction) ** 2)))

        print(
            f"  RMSE={rmse:.4e} | ECE={ece:.4f} | Coverage@90={coverage_90:.3f} "
            f"| UQ-error corr={correlation:.3f}"
        )

        all_results.append(
            {
                "name": analog.name,
                "mass_msun": analog.virial_mass_msun,
                "concentration": analog.concentration,
                "z_l": analog.z_l,
                "z_s": analog.z_s,
                "grid_size": args.grid,
                "extent_arcsec": analog.extent_arcsec,
                "rmse": rmse,
                "ece": ece,
                "confidence_levels": confidence_levels.tolist(),
                "coverages": coverages.tolist(),
                "coverage_90": coverage_90,
                "uq_error_correlation": correlation,
                "pixel_uncertainties": pixel_uncertainties,
                "pixel_errors": pixel_errors,
            }
        )

    mean_ece = float(np.mean([result["ece"] for result in all_results]))
    mean_coverage_90 = float(np.mean([result["coverage_90"] for result in all_results]))
    mean_corr = float(np.mean([result["uq_error_correlation"] for result in all_results]))
    mean_rmse = float(np.mean([result["rmse"] for result in all_results]))

    print(f"\n{'=' * 79}")
    print("  SUMMARY")
    print(f"  Mean held-out RMSE:       {mean_rmse:.4e}")
    print(f"  Mean ECE:                {mean_ece:.4f}")
    print(f"  Mean coverage @ 90%:     {mean_coverage_90:.3f}")
    print(f"  Mean UQ-error corr:      {mean_corr:.3f}")
    print(f"  Validation scale factor: {bundle.uncertainty_scale:.3f}")
    print(f"{'=' * 79}")

    _generate_calibration_figure(all_results, outdir)

    json_results = [
        {
            key: value
            for key, value in result.items()
            if key not in {"pixel_uncertainties", "pixel_errors"}
        }
        for result in all_results
    ]
    json_results.append(
        {
            "summary": {
                "mean_rmse": mean_rmse,
                "mean_ece": mean_ece,
                "mean_coverage_90": mean_coverage_90,
                "mean_uq_error_correlation": mean_corr,
                "prediction_mode": "checkpoint_backed_mc_dropout",
                "evaluation_mode": "synthetic_held_out_nfw_analogs",
                "publication_valid": True,
                "publication_scope": "synthetic NFW analog calibration only; not observational posterior calibration",
                "checkpoint_path": str(bundle.checkpoint_path),
                "checkpoint_calibration_ece": bundle.calibration_ece,
                "checkpoint_calibration_coverage_90": bundle.calibration_coverage_90,
                "calibrated_z_scores": bundle.calibrated_z_scores.tolist(),
                "calibration_shrinkage": bundle.calibration_shrinkage,
                "training_config": asdict(bundle.config),
            }
        }
    )
    json_path = outdir / "uncertainty_calibration_results.json"
    json_path.write_text(json.dumps(json_results, indent=2), encoding="utf-8")
    print(f"Calibration JSON saved: {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
