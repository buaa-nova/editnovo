import numpy as np
import matplotlib.pyplot as plt

# Configuration
warmup_iters = 10940  #
cosine_schedule_period_iters = 600_000
base_lr = 5e-4

# Generate iteration steps
steps = np.arange(0, cosine_schedule_period_iters + 1)

# Compute lr factor for each step
def get_lr_factor(epoch):
    lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / cosine_schedule_period_iters))
    if epoch <= warmup_iters:
        lr_factor *= epoch / warmup_iters
    return lr_factor

lr_factors = np.array([get_lr_factor(s) for s in steps])
lrs = base_lr * lr_factors

# Plot and save
plt.figure()
plt.plot(steps, lrs)
plt.xlabel("Iteration")
plt.ylabel("Learning Rate")
plt.title("Warm-up + Cosine Decay Learning Rate Schedule")
plt.grid(True)
plt.savefig('./lr_schedule.png')
plt.show()
