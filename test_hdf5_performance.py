#!/usr/bin/env python3
"""Test script to benchmark HDF5 random access performance."""

import time
import random
from casanovo.depthcharge.data.hdf5 import SpectrumIndex

def benchmark_random_access(index_path, num_tests=200, seed=42, preload=False):
    """Benchmark random access performance of SpectrumIndex.
    
    Parameters
    ----------
    index_path : str
        Path to the HDF5 index file
    num_tests : int
        Number of random access tests to perform
    seed : int
        Random seed for reproducible results
    preload : bool
        Whether to preload group data for better performance
    """
    random.seed(seed)
    
    print(f"Benchmarking random access performance on {index_path}")
    print(f"Number of tests: {num_tests}")
    print(f"Preload optimization: {preload}")
    
    try:
        with SpectrumIndex(index_path, annotated=True) as index:
            n_spectra = len(index)
            print(f"Total spectra in index: {n_spectra}")
            
            if n_spectra == 0:
                print("Index is empty, cannot benchmark")
                return
            
            # Generate random indices for testing
            test_indices = [random.randint(0, n_spectra - 1) for _ in range(num_tests)]
            
            if preload:
                print("Preloading group data...")
                preload_start = time.time()
                index._preload_group_data()
                preload_time = time.time() - preload_start
                print(f"Preload time: {preload_time:.3f} seconds")
            
            # Warm up - access a few spectra to initialize caches
            print("Warming up...")
            warmup_start = time.time()
            for i in range(min(10, n_spectra)):
                _ = index.get_spectrum(i)
            warmup_time = time.time() - warmup_start
            print(f"Warmup time: {warmup_time:.3f} seconds")
            
            # Benchmark random access
            print("Starting benchmark...")
            start_time = time.time()
            
            for idx in test_indices:
                spectrum = index.get_spectrum(idx)
                # Ensure we actually use the data to prevent optimization
                _ = len(spectrum[0])  # mz_array length
            
            end_time = time.time()
            
            total_time = end_time - start_time
            avg_time_per_access = total_time / num_tests
            
            print(f"Total time: {total_time:.3f} seconds")
            print(f"Average time per spectrum access: {avg_time_per_access*1000:.3f} ms")
            print(f"Spectra per second: {num_tests/total_time:.1f}")
            
    except Exception as e:
        print(f"Error during benchmark: {e}")
        import traceback
        traceback.print_exc()
        print("Make sure you have a valid HDF5 index file to test with")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        index_path = sys.argv[1]
    else:
        # Look for HDF5 files in common locations
        import glob
        hdf5_files = glob.glob("*.h5") + glob.glob("*.hdf5") + glob.glob("**/*.h5", recursive=True) + glob.glob("**/*.hdf5", recursive=True)
        
        if hdf5_files:
            index_path = hdf5_files[0]
            print(f"Found HDF5 file: {index_path}")
        else:
            print("No HDF5 files found. Please provide a path to an HDF5 index file.")
            print("Usage: python test_hdf5_performance.py <path_to_hdf5_file>")
            sys.exit(1)
    
    # Run without preloading first
    print("=== Testing WITHOUT preloading ===")
    benchmark_random_access(index_path, num_tests=50, preload=False)
    
    print("\n=== Testing WITH preloading ===")
    benchmark_random_access(index_path, num_tests=200, preload=True)
