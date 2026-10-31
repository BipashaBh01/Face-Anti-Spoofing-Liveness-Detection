# Import all necessary packages
from lib_imports import *
# from custom_module_imports import *
from Functions.config import *


"""
## Optimiser & Scheduler

It controls:

- How fast each part learns (learning rate)
- How learning changes over time (scheduler

The system is doing this:
- Start gently --> avoid instability
- Learn quickly --> capture patterns
- Slow down --> refine
- Protect pretrained knowledge
- Train new parts aggressively 
"""


#=========================================================
# MAIN CLASS FOR OPTIMISER AND SCHEDULER
#=========================================================

class WarmupCosineScheduler:
    """
    Custom Learning Rate Scheduler (not built in PyTorch): Linear Warmup + Cosine Decay

    Purpose:
    - Start training with a small learning rate (stabilises training)
    - Gradually increase LR during warmup phase
    - Then decay LR smoothly using cosine schedule

    Preserves per-group differential LRs:
    - Each parameter group keeps its OWN starting LR (e.g. the pretrained RGB
      backbone at 0.1x). The scheduler scales every group by the SAME factor,
      so the warmup/cosine *shape* is shared but the relative ratios are kept.

    Phases:
    1. Warmup: factor rises linearly 0 -> 1   (so each group: 0 -> its base LR)
    2. Cosine decay: factor falls 1 -> min/base smoothly

    Args:
        opt            : optimizer (e.g., AdamW) with per-group 'lr' values
        warmup_epochs  : number of warmup epochs
        total_epochs   : total training epochs
        base_lr        : peak (reference) learning rate used for the min ratio
        min_lr         : minimum learning rate floor (for the reference group)
    """

    def __init__(self, opt, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
        # Store optimiser reference
        self.opt = opt

        # Number of warmup epochs (linear increase phase)
        self.warmup = warmup_epochs

        # Total number of epochs
        self.total = total_epochs

        # Peak (reference) learning rate
        self.base = base_lr

        # Minimum learning rate after decay
        self.min = min_lr

        # Capture each group's STARTING lr, so we can scale them by a shared
        # factor and preserve their relative differences (e.g. backbone 0.1x).
        self.base_lrs = [pg['lr'] for pg in opt.param_groups]

    def step(self, epoch):
        """
        Update learning rate for the current epoch.

        Args:
            epoch : current epoch index (int)

        Returns:
            list of the updated LRs (one per parameter group, for logging)
        """

        # --- Phase 1: Warmup (linear increase) -------------------
        if epoch < self.warmup:
            # Factor rises 0 -> 1 across the warmup epochs.
            factor = (epoch + 1) / self.warmup

        # --- Phase 2: Cosine decay -------------------------------
        else:
            # Progress through the decay phase (0 -> 1)
            p = (epoch - self.warmup) / (self.total - self.warmup)

            # Cosine term goes 1 -> 0 smoothly
            cos = 0.5 * (1 + math.cos(math.pi * p))

            # Factor goes from 1 down to (min/base), so the reference group
            # decays base_lr -> min_lr, and every group follows proportionally.
            min_ratio = self.min / self.base
            factor = min_ratio + (1 - min_ratio) * cos

        # Apply the SAME factor to every group, scaling each group's OWN base LR.
        # This keeps the differential ratios (e.g. backbone stays 0.1x).
        for pg, base in zip(self.opt.param_groups, self.base_lrs):
            pg['lr'] = base * factor

        # Return current LRs (for logging)
        return [pg['lr'] for pg in self.opt.param_groups]

