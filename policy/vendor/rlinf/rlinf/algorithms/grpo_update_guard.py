# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# http://www.apache.org/licenses/LICENSE-2.0
"""Collective empty-minibatch guard for the independent PhysCanvas fork."""

from __future__ import annotations

import torch
import torch.distributed as dist


def has_global_valid_samples(
    loss_mask: torch.Tensor | None,
    *,
    local_batch_size: int,
    device: torch.device | str,
) -> bool:
    """Agree across actor ranks whether any loss-bearing element exists.

    Call exactly once on every actor rank before each global minibatch. Never
    skip FSDP work based on a local-only mask: other ranks may still be valid.
    In non-distributed CPU tests this is equivalent to the local decision.
    """
    if loss_mask is None:
        count = torch.tensor(int(local_batch_size), dtype=torch.long, device=device)
    else:
        count = loss_mask.detach().count_nonzero().to(device=device, dtype=torch.long)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
    return bool(count.item() > 0)
