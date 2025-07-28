#!/usr/bin/env python3
"""
Test script to visualize the new constant phase learning rate scheduler with cosine final decay.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import sys
import os

# Add the casanovo module to the path
sys.path.insert(0, '/root/attennovo')

from casanovo.denovo.model import CosineWarmupConstantScheduler

def test_constant_phase_scheduler():
    """Test the new constant phase learning rate scheduler."""
    
    # Create a dummy model and optimizer
    model = nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    
    # Create the scheduler with configurable parameters
    warmup_iters = 20_000
    cosine_schedule_period_iters = 100_000
    constant_lr_iters = 20_000
    final_decay_iters = 20_000
    constant_lr_factor = 0.08  # 8% of peak LR
    min_lr_factor = 0.001     # 0.1% of base LR
    
    scheduler = CosineWarmupConstantScheduler(
        optimizer, 
        warmup_iters=warmup_iters,
        cosine_schedule_period_iters=cosine_schedule_period_iters,
        constant_lr_iters=constant_lr_iters,
        final_decay_iters=final_decay_iters,
        constant_lr_factor=constant_lr_factor,
        min_lr_factor=min_lr_factor
    )
    
    # Test for multiple cycles
    num_iters = 200_000  # Test for 200k iterations to see multiple cycles
    lr_values = []
    
    for i in range(num_iters):
        lr_factor = scheduler.get_lr_factor(i)
        lr_values.append(lr_factor * 5e-4)  # Base LR is 5e-4

    # Plot the learning rate schedule
    plt.figure(figsize=(16, 10))
    plt.plot(range(num_iters), lr_values, linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Learning Rate')
    plt.title('Constant Phase Learning Rate Schedule\n(First: Warmup→Cosine→Constant→Decay, Subsequent: Constant→Decay)')
    plt.grid(True, alpha=0.3)
    
    # Add vertical lines to show phase boundaries
    first_cycle_phases = [
        (warmup_iters, 'Warmup End', 'green'),
        (warmup_iters + cosine_schedule_period_iters, 'Cosine End', 'blue'),
        (warmup_iters + cosine_schedule_period_iters + constant_lr_iters, 'Constant End', 'orange'),
        (warmup_iters + cosine_schedule_period_iters + constant_lr_iters + final_decay_iters, 'First Cycle End', 'red')
    ]
    
    for iter_pos, label, color in first_cycle_phases:
        if iter_pos < num_iters:
            plt.axvline(x=iter_pos, color=color, linestyle='--', alpha=0.7, label=label)
    
    # Add cycle boundaries for subsequent cycles
    first_cycle_total = warmup_iters + cosine_schedule_period_iters + constant_lr_iters + final_decay_iters
    subsequent_cycle_length = constant_lr_iters + final_decay_iters
    
    for cycle in range(1, 4):  # Show next 3 cycles
        cycle_start = first_cycle_total + (cycle - 1) * subsequent_cycle_length
        cycle_end = first_cycle_total + cycle * subsequent_cycle_length
        
        if cycle_start < num_iters:
            plt.axvline(x=cycle_start, color='purple', linestyle=':', alpha=0.5, 
                       label=f'Cycle {cycle+1} Start' if cycle == 1 else "")
        if cycle_end < num_iters:
            plt.axvline(x=cycle_end, color='red', linestyle=':', alpha=0.5,
                       label=f'Cycle {cycle+1} End' if cycle == 1 else "")
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('/root/attennovo/constant_phase_lr_visualization.png', dpi=300, bbox_inches='tight')
    print("Constant phase learning rate schedule visualization saved to: /root/attennovo/constant_phase_lr_visualization.png")
    
    # Print detailed analysis
    print(f"\nConstant Phase Learning Rate Schedule Analysis:")
    print(f"Base learning rate: 5e-4")
    print(f"Constant LR factor: {constant_lr_factor} ({constant_lr_factor*100}% of peak)")
    print(f"Min LR factor: {min_lr_factor} ({min_lr_factor*100}% of base)")
    print()
    
    print("First Cycle Phases:")
    print(f"  1. Warmup: 0 → {warmup_iters:,} iterations")
    print(f"  2. Cosine decay: {warmup_iters:,} → {warmup_iters + cosine_schedule_period_iters:,} iterations")
    print(f"  3. Constant LR: {warmup_iters + cosine_schedule_period_iters:,} → {warmup_iters + cosine_schedule_period_iters + constant_lr_iters:,} iterations")
    print(f"  4. Final decay: {warmup_iters + cosine_schedule_period_iters + constant_lr_iters:,} → {first_cycle_total:,} iterations")
    print(f"  Total first cycle: {first_cycle_total:,} iterations")
    print()
    
    print("Subsequent Cycles:")
    print(f"  1. Constant LR: {constant_lr_iters:,} iterations")
    print(f"  2. Final decay: {final_decay_iters:,} iterations")  
    print(f"  Total per cycle: {subsequent_cycle_length:,} iterations")
    print()
    
    # Show key LR values
    base_lr = 5e-4
    peak_lr = scheduler.get_lr_factor(warmup_iters) * base_lr
    constant_lr = scheduler.get_lr_factor(warmup_iters + cosine_schedule_period_iters + 1000) * base_lr
    min_lr = scheduler.get_lr_factor(first_cycle_total - 1) * base_lr
    
    print("Key Learning Rate Values:")
    print(f"  Peak LR (end of warmup): {peak_lr:.2e}")
    print(f"  Constant LR: {constant_lr:.2e}")
    print(f"  Min LR: {min_lr:.2e}")
    print()
    
    # Validate the ratios
    print("Validation:")
    constant_ratio = constant_lr / peak_lr
    min_ratio = min_lr / base_lr
    print(f"  Constant/Peak ratio: {constant_ratio:.3f} (expected: {constant_lr_factor})")
    print(f"  Min/Base ratio: {min_ratio:.3f} (expected: {min_lr_factor})")
    
    assert abs(constant_ratio - constant_lr_factor) < 0.01, f"Constant LR ratio mismatch"
    assert abs(min_ratio - min_lr_factor) < 0.01, f"Min LR ratio mismatch"
    
    print("  ✅ All ratios correct!")

def compare_linear_vs_cosine_decay():
    """Compare the old linear decay with the new cosine decay in the final phase."""
    
    print("\n" + "="*60)
    print("COMPARING LINEAR vs COSINE FINAL DECAY")
    print("="*60)
    
    # Parameters for the final decay phase
    constant_lr_factor = 0.1
    min_lr_factor = 0.001
    final_decay_iters = 1000
    
    # Generate decay curves
    progress = np.linspace(0, 1, final_decay_iters)
    
    # Linear decay (old behavior)
    linear_decay = constant_lr_factor - (constant_lr_factor - min_lr_factor) * progress
    
    # Cosine decay (new behavior)
    cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))
    cosine_decay = min_lr_factor + (constant_lr_factor - min_lr_factor) * cosine_factor
    
    # Create comparison plot
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(progress * 100, linear_decay, 'r-', linewidth=2, label='Linear Decay (Old)')
    plt.plot(progress * 100, cosine_decay, 'b-', linewidth=2, label='Cosine Decay (New)')
    plt.axhline(y=constant_lr_factor, color='gray', linestyle='--', alpha=0.7, label='Start (Constant LR)')
    plt.axhline(y=min_lr_factor, color='gray', linestyle='--', alpha=0.7, label='End (Min LR)')
    plt.xlabel('Decay Progress (%)')
    plt.ylabel('LR Factor')
    plt.title('Final Decay Phase: Linear vs Cosine Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Show the difference
    plt.subplot(2, 1, 2)
    difference = cosine_decay - linear_decay
    plt.plot(progress * 100, difference, 'g-', linewidth=2)
    plt.xlabel('Decay Progress (%)')
    plt.ylabel('LR Factor Difference (Cosine - Linear)')
    plt.title('Difference Between Cosine and Linear Decay')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('/root/attennovo/linear_vs_cosine_decay_comparison.png', dpi=300, bbox_inches='tight')
    print("Decay comparison visualization saved to: /root/attennovo/linear_vs_cosine_decay_comparison.png")
    
    # Analyze key differences
    print(f"\nKey Differences Analysis:")
    print(f"Constant LR factor: {constant_lr_factor}")
    print(f"Min LR factor: {min_lr_factor}")
    print(f"Decay iterations: {final_decay_iters}")
    print()
    
    # Compare at specific points
    points = [0.25, 0.5, 0.75]
    for p in points:
        idx = int(p * final_decay_iters)
        linear_val = linear_decay[idx]
        cosine_val = cosine_decay[idx]
        diff = cosine_val - linear_val
        print(f"At {p*100}% progress:")
        print(f"  Linear: {linear_val:.6f}")
        print(f"  Cosine: {cosine_val:.6f}")
        print(f"  Difference: {diff:.6f} ({diff/linear_val*100:+.1f}%)")
        print()
    
    # Summary
    max_diff = np.max(np.abs(difference))
    max_diff_idx = np.argmax(np.abs(difference))
    max_diff_progress = progress[max_diff_idx] * 100
    
    print(f"Maximum difference: {max_diff:.6f} at {max_diff_progress:.1f}% progress")
    print(f"Cosine decay provides {'smoother' if max_diff > 0 else 'steeper'} initial decay")

def test_subsequent_constant_lr_ratio():
    """Test the subsequent_constant_lr_ratio parameter."""
    
    print("\n" + "="*60)
    print("TESTING SUBSEQUENT CONSTANT LR RATIO PARAMETER")
    print("="*60)
    
    # Create dummy optimizer
    model = nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Test parameters
    warmup_iters = 5_000
    cosine_schedule_period_iters = 20_000  # First cycle cosine decay period
    constant_lr_iters = 8_000              # First cycle constant LR duration
    final_decay_iters = 2_000
    
    # Test different ratios
    test_ratios = [0.5, 1.0, 1.5, 2.0]
    
    plt.figure(figsize=(15, 10))
    
    for i, ratio in enumerate(test_ratios):
        print(f"\nTesting subsequent_constant_lr_ratio = {ratio}")
        
        scheduler = CosineWarmupConstantScheduler(
            optimizer=optimizer,
            warmup_iters=warmup_iters,
            cosine_schedule_period_iters=cosine_schedule_period_iters,
            constant_lr_iters=constant_lr_iters,
            final_decay_iters=final_decay_iters,
            subsequent_constant_lr_ratio=ratio
        )
        
        # Calculate expected subsequent constant LR duration
        expected_subsequent_duration = int(cosine_schedule_period_iters * ratio)
        print(f"  cosine_schedule_period_iters: {cosine_schedule_period_iters:,}")
        print(f"  ratio: {ratio}")
        print(f"  Expected subsequent constant LR duration: {expected_subsequent_duration:,}")
        print(f"  Actual subsequent_constant_lr_iters: {scheduler.subsequent_constant_lr_iters:,}")
        
        # Verify calculation
        assert scheduler.subsequent_constant_lr_iters == expected_subsequent_duration
        print(f"  ✅ Calculation correct!")
        
        # Generate schedule for visualization
        first_cycle_total = warmup_iters + cosine_schedule_period_iters + constant_lr_iters + final_decay_iters
        second_cycle_total = scheduler.subsequent_constant_lr_iters + final_decay_iters
        total_iters = first_cycle_total + second_cycle_total
        
        lr_factors = []
        for iter_num in range(total_iters):
            lr_factor = scheduler.get_lr_factor(iter_num)
            lr_factors.append(lr_factor)
        
        # Plot
        plt.subplot(2, 2, i + 1)
        plt.plot(lr_factors, linewidth=2, label=f'Ratio: {ratio}')
        plt.title(f'Subsequent Constant LR Ratio: {ratio}\n'
                 f'Subsequent Constant Duration: {expected_subsequent_duration:,} iters')
        plt.xlabel('Iteration')
        plt.ylabel('LR Factor')
        plt.grid(True, alpha=0.3)
        
        # Add phase markers
        plt.axvline(warmup_iters, color='r', linestyle='--', alpha=0.7, label='End Warmup')
        plt.axvline(warmup_iters + cosine_schedule_period_iters, color='g', linestyle='--', alpha=0.7, label='End Cosine')
        plt.axvline(warmup_iters + cosine_schedule_period_iters + constant_lr_iters, color='orange', linestyle='--', alpha=0.7, label='End 1st Constant')
        plt.axvline(first_cycle_total, color='purple', linestyle='--', alpha=0.7, label='End Cycle 1')
        plt.axvline(first_cycle_total + scheduler.subsequent_constant_lr_iters, color='brown', linestyle='--', alpha=0.7, label='End 2nd Constant')
        
        if i == 0:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('/root/attennovo/subsequent_constant_lr_test.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: /root/attennovo/subsequent_constant_lr_test.png")
    
    # Summary
    print(f"\n" + "="*50)
    print("SUMMARY:")
    print(f"✅ subsequent_constant_lr_ratio parameter works correctly!")
    print(f"✅ Subsequent constant LR duration = ratio × cosine_schedule_period_iters")
    print(f"✅ cosine_schedule_period_iters = {cosine_schedule_period_iters:,}")
    print(f"✅ Examples:")
    for ratio in test_ratios:
        duration = int(cosine_schedule_period_iters * ratio)
        print(f"   - ratio {ratio} → {duration:,} iterations ({ratio:.0%} of cosine period)")

if __name__ == "__main__":
    test_constant_phase_scheduler()
    compare_linear_vs_cosine_decay()
    test_subsequent_constant_lr_ratio()
