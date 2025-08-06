#!/usr/bin/env python3
"""Test script to benchmark HDF5 batch access performance for 512 batch size."""

import time
import random
import numpy as np
from casanovo.depthcharge.data.hdf5 import AnnotatedSpectrumIndex

def benchmark_batch_access(index_path, batch_size=512, num_batches=20, seed=42):
    """Benchmark batch access performance to reach 10 batches/s target.
    
    Parameters
    ----------
    index_path : str
        Path to the HDF5 index file
    batch_size : int
        Number of spectra per batch (default 512)
    num_batches : int
        Number of batches to test
    seed : int
        Random seed for reproducible results
    """
    random.seed(seed)
    np.random.seed(seed)
    
    print(f"🚀 HDF5 BATCH PERFORMANCE BENCHMARK")
    print("=" * 50)
    print(f"Index file: {index_path}")
    print(f"Batch size: {batch_size}")
    print(f"Number of batches: {num_batches}")
    print(f"Target: 10 batches/s (5120 spectra/s)")
    print()
    
    try:
        with AnnotatedSpectrumIndex(index_path) as index:
            n_spectra = len(index)
            print(f"Total spectra in index: {n_spectra:,}")
            
            if n_spectra < batch_size:
                print(f"❌ Index has fewer spectra ({n_spectra}) than batch size ({batch_size})")
                return
            
            # Test 1: Individual spectrum access (baseline)
            print("📊 Test 1: Individual Spectrum Access (Baseline)")
            print("-" * 50)
            
            # Generate random indices for one batch
            single_batch_indices = np.random.choice(n_spectra, size=batch_size, replace=False)
            
            start_time = time.time()
            individual_spectra = []
            
            for idx in single_batch_indices:
                spectrum_data = index.get_spectrum(idx)
                individual_spectra.append(spectrum_data)
            
            individual_time = time.time() - start_time
            individual_rate = batch_size / individual_time
            
            print(f"Time for {batch_size} individual accesses: {individual_time:.3f}s")
            print(f"Individual access rate: {individual_rate:.1f} spectra/s")
            print()
            
            # Test 2: Sequential batch access
            print("🏃 Test 2: Sequential Batch Access")
            print("-" * 50)
            
            start_time = time.time()
            total_spectra = 0
            batch_times = []
            
            for batch_idx in range(num_batches):
                batch_start_time = time.time()
                
                # Generate sequential indices for this batch
                start_idx = (batch_idx * batch_size) % (n_spectra - batch_size)
                batch_indices = list(range(start_idx, start_idx + batch_size))
                
                # Access spectra in this batch
                batch_spectra = []
                for idx in batch_indices:
                    spectrum_data = index.get_spectrum(idx)
                    batch_spectra.append(spectrum_data)
                
                batch_time = time.time() - batch_start_time
                batch_times.append(batch_time)
                total_spectra += len(batch_spectra)
                
                batch_rate = 1.0 / batch_time if batch_time > 0 else 0
                spectra_rate = batch_size / batch_time if batch_time > 0 else 0
                
                print(f"  Batch {batch_idx:2d}: {batch_size} spectra, {batch_rate:.2f} batches/s, {spectra_rate:.0f} spectra/s")
            
            sequential_total_time = time.time() - start_time
            sequential_batch_rate = num_batches / sequential_total_time
            sequential_spectra_rate = total_spectra / sequential_total_time
            
            print(f"\nSequential Results:")
            print(f"  Total time: {sequential_total_time:.3f}s")
            print(f"  Average batch rate: {sequential_batch_rate:.2f} batches/s")
            print(f"  Average spectra rate: {sequential_spectra_rate:.0f} spectra/s")
            print()
            
            # Test 3: Random batch access
            print("🎲 Test 3: Random Batch Access")
            print("-" * 50)
            
            start_time = time.time()
            total_spectra = 0
            batch_times = []
            
            for batch_idx in range(num_batches):
                batch_start_time = time.time()
                
                # Generate random indices for this batch
                batch_indices = np.random.choice(n_spectra, size=batch_size, replace=False)
                
                # Access spectra in this batch
                batch_spectra = []
                for idx in batch_indices:
                    spectrum_data = index.get_spectrum(idx)
                    batch_spectra.append(spectrum_data)
                
                batch_time = time.time() - batch_start_time
                batch_times.append(batch_time)
                total_spectra += len(batch_spectra)
                
                batch_rate = 1.0 / batch_time if batch_time > 0 else 0
                spectra_rate = batch_size / batch_time if batch_time > 0 else 0
                
                print(f"  Batch {batch_idx:2d}: {batch_size} spectra, {batch_rate:.2f} batches/s, {spectra_rate:.0f} spectra/s")
            
            random_total_time = time.time() - start_time
            random_batch_rate = num_batches / random_total_time
            random_spectra_rate = total_spectra / random_total_time
            
            print(f"\nRandom Results:")
            print(f"  Total time: {random_total_time:.3f}s")
            print(f"  Average batch rate: {random_batch_rate:.2f} batches/s")
            print(f"  Average spectra rate: {random_spectra_rate:.0f} spectra/s")
            print()
            
            # Test 4: Optimized grouped access (simulate batch optimization)
            print("⚡ Test 4: Optimized Grouped Access")
            print("-" * 50)
            
            start_time = time.time()
            total_spectra = 0
            batch_times = []
            
            for batch_idx in range(num_batches):
                batch_start_time = time.time()
                
                # Generate indices and sort them for better HDF5 access patterns
                batch_indices = np.random.choice(n_spectra, size=batch_size, replace=False)
                batch_indices = np.sort(batch_indices)  # Sort for better access pattern
                
                # Access spectra in this batch (simulating optimized access)
                batch_spectra = []
                for idx in batch_indices:
                    spectrum_data = index.get_spectrum(idx)
                    batch_spectra.append(spectrum_data)
                
                batch_time = time.time() - batch_start_time
                batch_times.append(batch_time)
                total_spectra += len(batch_spectra)
                
                batch_rate = 1.0 / batch_time if batch_time > 0 else 0
                spectra_rate = batch_size / batch_time if batch_time > 0 else 0
                
                print(f"  Batch {batch_idx:2d}: {batch_size} spectra, {batch_rate:.2f} batches/s, {spectra_rate:.0f} spectra/s")
            
            optimized_total_time = time.time() - start_time
            optimized_batch_rate = num_batches / optimized_total_time
            optimized_spectra_rate = total_spectra / optimized_total_time
            
            print(f"\nOptimized Results:")
            print(f"  Total time: {optimized_total_time:.3f}s")
            print(f"  Average batch rate: {optimized_batch_rate:.2f} batches/s")
            print(f"  Average spectra rate: {optimized_spectra_rate:.0f} spectra/s")
            print()
            
            # Summary and target evaluation
            print("=" * 50)
            print("📈 PERFORMANCE SUMMARY")
            print("=" * 50)
            
            target_batch_rate = 10.0
            target_spectra_rate = target_batch_rate * batch_size
            
            results = [
                ("Individual access", individual_rate / batch_size, individual_rate),
                ("Sequential batches", sequential_batch_rate, sequential_spectra_rate),
                ("Random batches", random_batch_rate, random_spectra_rate),
                ("Optimized batches", optimized_batch_rate, optimized_spectra_rate),
            ]
            
            print(f"{'Method':<20} | {'Batches/s':<10} | {'Spectra/s':<12} | {'Target'}")
            print("-" * 60)
            
            best_method = None
            best_rate = 0
            
            for method, batch_rate, spectra_rate in results:
                status = "✅" if batch_rate >= target_batch_rate else "❌"
                print(f"{method:<20} | {batch_rate:<10.2f} | {spectra_rate:<12.0f} | {status}")
                
                if batch_rate > best_rate:
                    best_rate = batch_rate
                    best_method = method
            
            print()
            print(f"🎯 TARGET ANALYSIS:")
            print(f"   Target: {target_batch_rate} batches/s ({target_spectra_rate} spectra/s)")
            print(f"   Best method: {best_method} ({best_rate:.2f} batches/s)")
            
            if best_rate >= target_batch_rate:
                print(f"   ✅ TARGET ACHIEVED ({best_rate/target_batch_rate:.2f}x)")
            else:
                needed_speedup = target_batch_rate / best_rate
                print(f"   ❌ Need {needed_speedup:.2f}x speedup to reach target")
                print(f"   💡 Consider: batch HDF5 reads, caching, or async I/O")
            
            return best_rate
            
    except Exception as e:
        print(f"❌ Error during benchmark: {e}")
        import traceback
        traceback.print_exc()
        return 0

def benchmark_dataset_integration(index_path, batch_size=512, num_batches=10):
    """Test integration with AnnotatedSpectrumDataset - both original and optimized."""
    print("\n🔗 DATASET INTEGRATION TEST")
    print("=" * 50)
    
    try:
        from casanovo.data.datasets import AnnotatedSpectrumDataset
        from casanovo.denovo.dataloaders import prepare_batch
        from torch.utils.data import DataLoader
        
        with AnnotatedSpectrumIndex(index_path) as index:
            print(f"Dataset size: {len(index):,}")
            print(f"Batch size: {batch_size}")
            print(f"Test batches: {num_batches}")
            print()
            
            # Test 1: Original processing
            print("📊 Test 1: Original Dataset Processing")
            print("-" * 40)
            
            dataset_original = AnnotatedSpectrumDataset(
                index,
                n_peaks=150,
                min_mz=140.0,
                max_mz=2500.0,
                min_intensity=0.01,
                remove_precursor_tol=2.0,
                use_optimized_processing=False,  # Use original processing
            )
            
            dataloader_original = DataLoader(
                dataset_original,
                batch_size=batch_size,
                shuffle=True,
                num_workers=8,
                collate_fn=prepare_batch,
                pin_memory=True,
            )
            
            start_time = time.time()
            total_spectra_original = 0
            
            for i, (spectra, precursors, annotations) in enumerate(dataloader_original):
                if i >= num_batches:
                    break
                
                batch_size_actual = spectra.shape[0]
                total_spectra_original += batch_size_actual
                
                elapsed = time.time() - start_time
                batch_rate = (i + 1) / elapsed if elapsed > 0 else 0
                spectra_rate = total_spectra_original / elapsed if elapsed > 0 else 0
                
                print(f"  Batch {i}: {batch_size_actual} spectra, {batch_rate:.2f} batches/s, {spectra_rate:.0f} spectra/s")
            
            original_time = time.time() - start_time
            original_batch_rate = num_batches / original_time if original_time > 0 else 0
            original_spectra_rate = total_spectra_original / original_time if original_time > 0 else 0
            
            print(f"\nOriginal Processing Results:")
            print(f"  Total time: {original_time:.3f}s")
            print(f"  Final batch rate: {original_batch_rate:.2f} batches/s")
            print(f"  Final spectra rate: {original_spectra_rate:.0f} spectra/s")
            print()
            
            # Test 2: Optimized processing
            print("🚀 Test 2: Optimized Dataset Processing")
            print("-" * 40)
            
            dataset_optimized = AnnotatedSpectrumDataset(
                index,
                n_peaks=150,
                min_mz=140.0,
                max_mz=2500.0,
                min_intensity=0.01,
                remove_precursor_tol=2.0,
                use_optimized_processing=True,  # Use optimized processing
            )
            
            dataloader_optimized = DataLoader(
                dataset_optimized,
                batch_size=batch_size,
                shuffle=True,
                num_workers=8,
                collate_fn=prepare_batch,
                pin_memory=True,
            )
            
            start_time = time.time()
            total_spectra_optimized = 0
            
            for i, (spectra, precursors, annotations) in enumerate(dataloader_optimized):
                if i >= num_batches:
                    break
                
                batch_size_actual = spectra.shape[0]
                total_spectra_optimized += batch_size_actual
                
                elapsed = time.time() - start_time
                batch_rate = (i + 1) / elapsed if elapsed > 0 else 0
                spectra_rate = total_spectra_optimized / elapsed if elapsed > 0 else 0
                
                print(f"  Batch {i}: {batch_size_actual} spectra, {batch_rate:.2f} batches/s, {spectra_rate:.0f} spectra/s")
            
            optimized_time = time.time() - start_time
            optimized_batch_rate = num_batches / optimized_time if optimized_time > 0 else 0
            optimized_spectra_rate = total_spectra_optimized / optimized_time if optimized_time > 0 else 0
            
            print(f"\nOptimized Processing Results:")
            print(f"  Total time: {optimized_time:.3f}s")
            print(f"  Final batch rate: {optimized_batch_rate:.2f} batches/s")
            print(f"  Final spectra rate: {optimized_spectra_rate:.0f} spectra/s")
            print()
            
            # Compare results
            print("📈 Dataset Processing Comparison:")
            print("-" * 40)
            speedup = optimized_batch_rate / original_batch_rate if original_batch_rate > 0 else 0
            print(f"Original processing:  {original_batch_rate:.2f} batches/s ({original_spectra_rate:.0f} spectra/s)")
            print(f"Optimized processing: {optimized_batch_rate:.2f} batches/s ({optimized_spectra_rate:.0f} spectra/s)")
            print(f"Speedup: {speedup:.2f}x")
            
            # Check target achievement
            target_rate = 10.0
            if optimized_batch_rate >= target_rate:
                print(f"✅ TARGET ACHIEVED! ({optimized_batch_rate:.2f} batches/s)")
            else:
                needed = target_rate / optimized_batch_rate if optimized_batch_rate > 0 else float('inf')
                print(f"❌ Still need {needed:.2f}x more speedup")
            
            return optimized_batch_rate
            
    except ImportError as e:
        print(f"⚠️ Dataset integration test skipped: {e}")
        return 0
    except Exception as e:
        print(f"❌ Dataset integration error: {e}")
        import traceback
        traceback.print_exc()
        return 0

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        index_path = sys.argv[1]
    else:
        # Default path for the HDF5 file
        index_path = "/mnt/hdf5/cfbae77b2a5b4d60b93589eeeaa5abac.hdf5"
        print(f"Using default HDF5 file: {index_path}")
    
    print("🧪 HDF5 BATCH PERFORMANCE OPTIMIZATION TEST")
    print("=" * 60)
    print("Testing if HDF5 access can achieve 10 batches/s (5120 spectra/s)")
    print()
    
    # Run batch performance benchmark
    best_batch_rate = benchmark_batch_access(index_path, batch_size=512, num_batches=20)
    
    # Test dataset integration
    dataset_batch_rate = benchmark_dataset_integration(index_path, batch_size=512, num_batches=10)
    
    # Final summary
    print("\n" + "=" * 60)
    print("🎯 FINAL PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    target_rate = 10.0
    print(f"Target: {target_rate} batches/s")
    print(f"Best HDF5 performance: {best_batch_rate:.2f} batches/s")
    
    if dataset_batch_rate > 0:
        print(f"Dataset integration: {dataset_batch_rate:.2f} batches/s")
        overall_best = max(best_batch_rate, dataset_batch_rate)
    else:
        overall_best = best_batch_rate
    
    if overall_best >= target_rate:
        print(f"✅ SUCCESS: Target achieved ({overall_best:.2f} batches/s)")
        print("🚀 The HDF5 access is fast enough for the required performance!")
    else:
        speedup_needed = target_rate / overall_best if overall_best > 0 else float('inf')
        print(f"❌ OPTIMIZATION NEEDED: {speedup_needed:.2f}x speedup required")
        print("💡 Suggestions:")
        print("   - Implement batch HDF5 reads")
        print("   - Add metadata caching")
        print("   - Use memory mapping")
        print("   - Consider async I/O")
    
    print("=" * 60)
