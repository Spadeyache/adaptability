"""Synthetic CUDA compatibility check; no experiment traces or checkpoints."""
import json
import time
from pathlib import Path

from . import models
import torch


def main():
    device = models.configure_torch('cuda')
    records = []
    for variant in models.VARIANTS:
        torch.manual_seed(9183)
        model = models.DynamicsModel(variant).to(device)
        model.train()
        x = torch.randn(16, 15, device=device)
        a = torch.randn(16, 3, device=device)
        context = torch.randn(16, 2, device=device) if variant == 'oracle' else None
        history = torch.randn(16, 10, 33, device=device) if variant == 'history' else None
        z = model.embedding(torch.arange(16, device=device)) if model.z_dim else None
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        prediction = model(x, a, context=context, history=history, z=z)
        loss = prediction.square().mean()
        if model.z_dim:
            loss = loss + 1e-4 * model.embedding.weight.square().mean()
        assert bool(torch.isfinite(loss))
        loss.backward()
        assert all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
        optimizer.step()
        torch.cuda.synchronize()
        assert all(bool(torch.isfinite(p).all()) for p in model.parameters())
        records.append(dict(variant=variant, forward_backward_optimizer='passed'))
    result = dict(status='passed', synthetic_only=True, experiment_traces_read=False,
                  experiment_checkpoints_saved=False, torch_version=torch.__version__,
                  gpu=torch.cuda.get_device_name(device), variants=records,
                  model_source_sha256=models.digest(Path(models.__file__)),
                  checked_unix_seconds=time.time())
    models.write_new(models.ROOT/'results/gpu_preflight.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
