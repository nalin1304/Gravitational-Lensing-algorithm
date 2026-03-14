# Statistical Rigor Report

Generated: 2026-03-14T12:57:26.170930+00:00

## SLACS Validation Summary
- Systems: 5
- Joint pass rate: 100.00%
- Validation scopes: image_space_forward_model
- rmse: mean=0.100682, 95% CI=[0.093410, 0.107924]
- ssim: mean=0.985421, 95% CI=[0.982776, 0.987887]
- ring_correlation: mean=0.916368, 95% CI=[0.899521, 0.931899]
- annular_flux_ratio: mean=0.954093, 95% CI=[0.944823, 0.963363]

## Ablation Effect Summary
- Full pipeline RMSE mean: 0.043050
- Full pipeline pass rate: 33.33%
- RMSE gain vs no calibration: 0.170789
- RMSE gain vs vanilla: 0.000014

## SOTA Table Summary
- Methods compared: 4
- Our RMSE rank: 3
- Our learned-model RMSE rank: 1

## Uncertainty Calibration Summary
- Systems: 5
- Prediction mode: checkpoint_backed_mc_dropout
- Evaluation mode: synthetic_held_out_nfw_analogs
- Mean ECE: 0.062045
- Coverage@90%: 0.936426
- Publication scope: synthetic NFW analog calibration only; not observational posterior calibration

## Warnings
- None
