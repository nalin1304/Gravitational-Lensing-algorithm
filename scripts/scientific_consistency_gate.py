#!/usr/bin/env python3
import json
import numpy as np
import argparse
from typing import Dict, Any, Tuple

# Constants for empirical scaling laws
# Log Mass-to-Light ratio vs Total Mass (e.g., Faber-Jackson / Fundamental Plane proxies)
# Let's assume an empirical relation M/L approx 5.0 for typical massive ellipticals in solar units
# Range typically 2 to 10.
ML_MEAN = 5.0
ML_SIGMA = 2.0

# Dark Matter Fraction f_DM(R_eff) typical for massive ellipticals
# Typically 40% - 60% within the effective radius.
FDM_MEAN = 0.50
FDM_SIGMA = 0.15

class ScientificValidator:
    """
    Validates inferred lens model parameters against empirical astrophysical scaling laws.
    """
    
    def __init__(self, tolerance_sigma: float = 3.0):
        """
        :param tolerance_sigma: The number of standard deviations allowed before rejecting a model.
        """
        self.tolerance_sigma = tolerance_sigma
        
    def check_mass_to_light(self, total_mass_msun: float, total_luminosity_lsun: float) -> Tuple[bool, float, str]:
        """
        Validates the Mass-to-Light ratio.
        """
        if total_luminosity_lsun <= 0:
            return False, 0.0, "Total luminosity must be > 0"
            
        ml_ratio = total_mass_msun / total_luminosity_lsun
        
        lower_bound = max(0.1, ML_MEAN - self.tolerance_sigma * ML_SIGMA)
        upper_bound = ML_MEAN + self.tolerance_sigma * ML_SIGMA
        
        is_valid = lower_bound <= ml_ratio <= upper_bound
        msg = f"M/L ratio {ml_ratio:.2f} is outside the {self.tolerance_sigma}sigma empirical bounds [{lower_bound:.2f}, {upper_bound:.2f}]" if not is_valid else "M/L ratio is physically consistent."
        
        return is_valid, ml_ratio, msg
        
    def check_dark_matter_fraction(self, dm_mass_msun: float, stellar_mass_msun: float) -> Tuple[bool, float, str]:
        """
        Validates the Dark Matter fraction within a specific physical aperture.
        """
        total_mass = dm_mass_msun + stellar_mass_msun
        if total_mass <= 0:
            return False, 0.0, "Total mass must be > 0"
            
        f_dm = dm_mass_msun / total_mass
        
        lower_bound = max(0.0, FDM_MEAN - self.tolerance_sigma * FDM_SIGMA)
        upper_bound = min(1.0, FDM_MEAN + self.tolerance_sigma * FDM_SIGMA)
        
        is_valid = lower_bound <= f_dm <= upper_bound
        msg = f"DM Fraction {f_dm:.2f} is outside the {self.tolerance_sigma}sigma empirical bounds [{lower_bound:.2f}, {upper_bound:.2f}]" if not is_valid else "DM Fraction is physically consistent."
        
        return is_valid, f_dm, msg
        
    def validate_inferred_model(self, inferred_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main validation entry point. 
        Expects a dictionary containing inferred masses and luminosities.
        """
        required_keys = ['total_mass', 'total_luminosity', 'dm_mass', 'stellar_mass']
        for k in required_keys:
            if k not in inferred_params:
                raise ValueError(f"Missing required parameter for scientific validation: {k}")
                
        ml_valid, ml_ratio, ml_msg = self.check_mass_to_light(
            inferred_params['total_mass'], 
            inferred_params['total_luminosity']
        )
        
        fdm_valid, f_dm, fdm_msg = self.check_dark_matter_fraction(
            inferred_params['dm_mass'], 
            inferred_params['stellar_mass']
        )
        
        passed = ml_valid and fdm_valid
        
        return {
            "passed": passed,
            "metrics": {
                "mass_to_light": ml_ratio,
                "dark_matter_fraction": f_dm
            },
            "flags": {
                "ml_flag": ml_msg,
                "fdm_flag": fdm_msg
            }
        }


def main():
    parser = argparse.ArgumentParser(description="Scientific Consistency Gate for Lens Modeling")
    parser.add_argument("--params_file", type=str, required=True, help="JSON file containing the inferred parameters.")
    parser.add_argument("--tolerance_sigma", type=float, default=3.0, help="Rejection threshold in sigmas.")
    
    args = parser.parse_args()
    
    with open(args.params_file, 'r') as f:
        inferred_params = json.load(f)
        
    validator = ScientificValidator(tolerance_sigma=args.tolerance_sigma)
    results = validator.validate_inferred_model(inferred_params)
    
    # Print results to stdout
    print(json.dumps(results, indent=2))
    
    # Exit with code 1 if failed to block pipeline
    if not results['passed']:
        exit(1)
    else:
        exit(0)

if __name__ == "__main__":
    main()

