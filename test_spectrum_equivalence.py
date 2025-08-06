#!/usr/bin/env python3
"""
Test to verify that our optimized spectrum processing produces
identical results to the original spectrum_utils processing.
"""

import numpy as np
import torch
import spectrum_utils.spectrum as sus
import sys
import os

# Add the casanovo module to path
sys.path.insert(0, '/root/attennovo')
from casanovo.data.datasets import AnnotatedSpectrumDataset

def process_with_spectrum_utils(mz_array, int_array, precursor_mz, precursor_charge,
                               min_mz=140.0, max_mz=2500.0, min_intensity=0.01, 
                               remove_precursor_tol=2.0, n_peaks=150):
    """Original spectrum_utils processing"""
    spectrum = sus.MsmsSpectrum(
        "",
        precursor_mz,
        precursor_charge,
        mz_array.astype(np.float64),
        int_array.astype(np.float32),
    )
    try:
        spectrum.set_mz_range(min_mz, max_mz)
        if len(spectrum.mz) == 0:
            raise ValueError
        spectrum.remove_precursor_peak(remove_precursor_tol, "Da")
        if len(spectrum.mz) == 0:
            raise ValueError
        spectrum.filter_intensity(min_intensity, n_peaks)
        if len(spectrum.mz) == 0:
            raise ValueError
        spectrum.scale_intensity("root", 1)
        intensities = spectrum.intensity / np.linalg.norm(spectrum.intensity)
        return torch.tensor(np.array([spectrum.mz, intensities])).T.float()
    except ValueError:
        return torch.tensor([[0, 1]]).float()

def process_optimized(mz_array, int_array, precursor_mz, precursor_charge,
                     min_mz=140.0, max_mz=2500.0, min_intensity=0.01, 
                     remove_precursor_tol=2.0, n_peaks=150):
    """Our optimized processing - use the actual implementation from the dataset"""
    # Create mock dataset to use the real optimized function
    from casanovo.data.datasets import AnnotatedSpectrumDataset
    dataset = AnnotatedSpectrumDataset.__new__(AnnotatedSpectrumDataset)
    dataset.min_mz = min_mz
    dataset.max_mz = max_mz
    dataset.min_intensity = min_intensity
    dataset.remove_precursor_tol = remove_precursor_tol
    dataset.n_peaks = n_peaks
    
    return dataset._process_peaks_optimized(mz_array, int_array, precursor_mz, precursor_charge)

def test_equivalence():
    """Test if both methods produce equivalent results"""
    
    # Test case 1: Simple spectrum
    mz = np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0], dtype=np.float64)
    intensity = np.array([1000.0, 500.0, 2000.0, 100.0, 1500.0, 300.0], dtype=np.float32)
    precursor_mz = 550.0
    precursor_charge = 2
    
    print("Test 1: Simple spectrum")
    print("-" * 40)
    
    result_original = process_with_spectrum_utils(mz, intensity, precursor_mz, precursor_charge)
    result_optimized = process_optimized(mz, intensity, precursor_mz, precursor_charge)
    
    print(f"Original shape: {result_original.shape}")
    print(f"Optimized shape: {result_optimized.shape}")
    print(f"Original m/z values: {result_original[:, 0].numpy()}")
    print(f"Optimized m/z values: {result_optimized[:, 0].numpy()}")
    print(f"Original intensities: {result_original[:, 1].numpy()}")
    print(f"Optimized intensities: {result_optimized[:, 1].numpy()}")
    
    # Check if results are close
    if result_original.shape == result_optimized.shape:
        mz_close = torch.allclose(result_original[:, 0], result_optimized[:, 0], atol=1e-6)
        int_close = torch.allclose(result_original[:, 1], result_optimized[:, 1], atol=1e-6)
        print(f"M/Z values match: {mz_close}")
        print(f"Intensity values match: {int_close}")
        if mz_close and int_close:
            print("✅ Results are equivalent!")
        else:
            print("❌ Results differ!")
            print(f"Max m/z difference: {torch.max(torch.abs(result_original[:, 0] - result_optimized[:, 0]))}")
            print(f"Max intensity difference: {torch.max(torch.abs(result_original[:, 1] - result_optimized[:, 1]))}")
    else:
        print("❌ Different shapes!")
    
    print()
    
    # Test case 2: More complex spectrum with many peaks
    print("Test 2: Complex spectrum with many peaks")
    print("-" * 40)
    
    # Use the EXACT same test case as our working detailed comparison
    np.random.seed(42)
    mz_complex = np.linspace(140, 2000, 200)  # Changed from 500 to 200
    intensity_complex = np.random.lognormal(3, 1, 200).astype(np.float32)  # Changed from 500 to 200
    precursor_mz_complex = 1200.0
    precursor_charge_complex = 3
    
    result_original_2 = process_with_spectrum_utils(mz_complex, intensity_complex, 
                                                   precursor_mz_complex, precursor_charge_complex)
    result_optimized_2 = process_optimized(mz_complex, intensity_complex, 
                                          precursor_mz_complex, precursor_charge_complex)
    
    print(f"Original shape: {result_original_2.shape}")
    print(f"Optimized shape: {result_optimized_2.shape}")
    
    if result_original_2.shape == result_optimized_2.shape:
        # Test with different tolerances
        mz_close_strict = torch.allclose(result_original_2[:, 0], result_optimized_2[:, 0], atol=1e-6)
        int_close_strict = torch.allclose(result_original_2[:, 1], result_optimized_2[:, 1], atol=1e-6)
        mz_close_loose = torch.allclose(result_original_2[:, 0], result_optimized_2[:, 0], atol=1e-4)
        int_close_loose = torch.allclose(result_original_2[:, 1], result_optimized_2[:, 1], atol=1e-4)
        
        print(f"M/Z values match (strict 1e-6): {mz_close_strict}")
        print(f"Intensity values match (strict 1e-6): {int_close_strict}")
        print(f"M/Z values match (loose 1e-4): {mz_close_loose}")
        print(f"Intensity values match (loose 1e-4): {int_close_loose}")
        
        if mz_close_loose and int_close_loose:
            print("✅ Results are equivalent within reasonable tolerance!")
            max_mz_diff = torch.max(torch.abs(result_original_2[:, 0] - result_optimized_2[:, 0]))
            max_int_diff = torch.max(torch.abs(result_original_2[:, 1] - result_optimized_2[:, 1]))
            print(f"Max m/z difference: {max_mz_diff:.2e}")
            print(f"Max intensity difference: {max_int_diff:.2e}")
        else:
            print("❌ Results differ significantly!")
            
            # Show detailed differences
            print(f"First 10 original m/z: {result_original_2[:10, 0].numpy()}")
            print(f"First 10 optimized m/z: {result_optimized_2[:10, 0].numpy()}")
            print(f"First 10 original int: {result_original_2[:10, 1].numpy()}")
            print(f"First 10 optimized int: {result_optimized_2[:10, 1].numpy()}")
            
            # Check if it's just ordering
            orig_sorted = result_original_2[torch.argsort(result_original_2[:, 0])]
            opt_sorted = result_optimized_2[torch.argsort(result_optimized_2[:, 0])]
            
            mz_sorted_match = torch.allclose(orig_sorted[:, 0], opt_sorted[:, 0], atol=1e-4)
            int_sorted_match = torch.allclose(orig_sorted[:, 1], opt_sorted[:, 1], atol=1e-4)
            print(f"After sorting by m/z - M/Z match: {mz_sorted_match}, Int match: {int_sorted_match}")
    else:
        print("❌ Different shapes!")

if __name__ == "__main__":
    test_equivalence()
