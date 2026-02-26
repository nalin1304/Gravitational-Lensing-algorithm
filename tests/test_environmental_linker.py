import pytest
import os
import numpy as np
import pandas as pd
from src.data.environmental_linker import EnvironmentalLinker

def test_environmental_linker_csv(tmp_path):
    """Test generating a synthetic CSV catalog and calculating external convergence."""
    # Synthetic target: RA=10.0, Dec=20.0, zl=0.5, zs=2.0
    linker = EnvironmentalLinker(main_lens_ra=10.0, main_lens_dec=20.0, main_lens_z=0.5, source_z=2.0)
    
    # Generate 3 nearby galaxies
    # 1. Very close, massive (zl=0.5, similar to main lens)
    # 2. Further away, less massive
    # 3. Background galaxy (should contribute 0 to main source kappa)
    
    data = {
        'ra': [10.0 + 5.0/3600.0, 10.0 - 20.0/3600.0, 10.0 + 10.0/3600.0],
        'dec': [20.0, 20.0, 20.0],
        'redshift': [0.5, 0.3, 2.5],
        'mass_msun': [1e13, 1e12, 1e14]
    }
    
    df = pd.DataFrame(data)
    csv_file = os.path.join(tmp_path, "mock_catalog.csv")
    df.to_csv(csv_file, index=False)
    
    # Compute kappa
    kappa_ext = linker.compute_kappa_ext(csv_file, format='csv')
    
    assert kappa_ext > 0.0
    assert kappa_ext < 1.0 # Should be a reasonably small perturbation
    
    # Let's test the individual points logic:
    # Galaxy 3 (z=2.5) is behind the source (z=2.0), so its Einstein radius should be 0
    theta_e_3 = linker._estimate_einstein_radius(mass_msun=1e14, lens_z=2.5)
    assert theta_e_3 == 0.0
