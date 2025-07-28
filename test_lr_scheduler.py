#!/usr/bin/env python3
"""
Test script to visualize the new multi-cycle learning rate scheduler.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import sys
import os

# Add the casanovo module to the path
sys.path.insert(0, '/root/attennovo')

from casanovo.denovo.model import CosineWarmupScheduler

def test_lr_scheduler():
    """Test the new multi-cycle learning rate scheduler."""
    
    # Create a dummy model and optimizer
    model = nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    
    # Create the scheduler with example parameters
    warmup_iters = 20_000
    cosine_schedule_period_iters = 100_000
    scheduler = CosineWarmupScheduler(
        optimizer, 
        warmup_iters=warmup_iters, 
        cosine_schedule_period_iters=cosine_schedule_period_iters,
        min_lr_factor=0.001
    )
    
    # Test for multiple cycles
    num_iters = 400_000  # Test for 100k iterations to see multiple cycles
    lr_values = []
    
    for i in range(num_iters):
        lr_factor = scheduler.get_lr_factor(i)
        lr_values.append(lr_factor * 5e-4)  # Base LR is 5e-4

    # Plot the learning rate schedule
    plt.figure(figsize=(15, 8))
    plt.plot(range(num_iters), lr_values, linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Learning Rate')
    plt.title('Multi-Cycle Learning Rate Schedule\n(First Cycle: Warmup + Decay, Subsequent Cycles: Decay Only at 0.1x Peak LR)')
    plt.grid(True, alpha=0.3)
    
    # Add vertical lines to show cycle boundaries
    cycle_boundaries = []
    iteration = 0
    
    for cycle in range(5):  # Show first 5 cycles
        if cycle == 0:
            # First cycle has warmup + decay
            cycle_end = iteration + warmup_iters + cosine_schedule_period_iters
        else:
            # Subsequent cycles have decay only
            cycle_end = iteration + cosine_schedule_period_iters
            
        if cycle_end > num_iters:
            break
        cycle_boundaries.append(cycle_end)
        plt.axvline(x=cycle_end, color='red', linestyle='--', alpha=0.7, 
                   label=f'Cycle {cycle+1} End' if cycle == 0 else "")
        
        iteration = cycle_end
        
    # Add vertical line to show end of warmup in first cycle
    plt.axvline(x=warmup_iters, color='green', linestyle=':', alpha=0.7, 
               label='Warmup End (First Cycle Only)')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig('/root/attennovo/lr_schedule_visualization.png', dpi=300, bbox_inches='tight')
    print("Learning rate schedule visualization saved to: /root/attennovo/lr_schedule_visualization.png")
    
    # Print some key points
    print(f"\nLearning Rate Schedule Analysis:")
    print(f"Warmup iterations (first cycle only): {warmup_iters}")
    print(f"Cosine decay iterations (all cycles): {cosine_schedule_period_iters}")
    print(f"Base learning rate: 5e-4")
    
    # Show peak LR for first few cycles
    first_cycle_length = warmup_iters + cosine_schedule_period_iters
    for cycle in range(4):
        if cycle == 0:
            # First cycle: warmup to peak
            peak_iter = warmup_iters
            peak_lr = lr_values[peak_iter]
            print(f"Cycle {cycle+1} - Peak LR at iteration {peak_iter}: {peak_lr:.2e} (after warmup)")
        else:
            # Subsequent cycles: start at 0.1x peak (no warmup)
            cycle_start = first_cycle_length + (cycle - 1) * cosine_schedule_period_iters
            if cycle_start < num_iters:
                start_lr = lr_values[cycle_start]
                print(f"Cycle {cycle+1} - Start LR at iteration {cycle_start}: {start_lr:.2e} (no warmup, 0.1x peak)")

if __name__ == "__main__":
    test_lr_scheduler()
