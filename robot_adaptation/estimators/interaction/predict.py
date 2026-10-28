"""Persistent, label-free inference across completed attempts."""
from __future__ import annotations

import numpy as np
import torch

from . import ARCHITECTURE
from .data import extract_snippets
from .model import InteractionEstimator


class InteractionPredictor:
    def __init__(self, checkpoint_path):
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if saved["architecture"] != ARCHITECTURE:
            raise ValueError("Unsupported interaction estimator architecture")
        self.model = InteractionEstimator(saved["hidden"])
        self.model.load_state_dict(saved["state_dict"])
        self.model.eval()
        self.normalization = saved["normalization"]
        self.alignment = saved["alignment"]
        if self.alignment != "observable":
            raise ValueError("Deployment inference requires observable contact alignment")
        self.reset()

    def reset(self):
        self.evidence_sum = torch.zeros(self.model.hidden)
        self.attempts = self.snippets = self.informative_attempts = 0

    def update_episode(self, data):
        """Consume only a completed attempt. No true parameters/IDs are needed."""
        snippets = extract_snippets(data, alignment=self.alignment)
        self.attempts += 1
        if snippets:
            embeddings = []
            norm = self.normalization
            with torch.inference_mode():
                for clip in snippets:
                    x = (clip["x"] - norm["mean"]) / norm["std"]
                    outcome = (clip["outcome"] - norm["outcome_mean"]) / norm["outcome_std"]
                    embeddings.append(self.model.encode(torch.tensor(x, dtype=torch.float32)[None],
                                      torch.tensor([len(x)]), torch.tensor(outcome, dtype=torch.float32)[None])[0])
            self.evidence_sum += torch.stack(embeddings).mean(dim=0)
            self.informative_attempts += 1
            self.snippets += len(snippets)
        return self.predict()

    def predict(self):
        norm = self.normalization
        if self.informative_attempts:
            with torch.inference_mode():
                log = self.model.predict_standard_log(self.evidence_sum / self.informative_attempts)
            physical_log = log.numpy() * norm["target_log_std"] + norm["target_log_mean"]
            if not np.isfinite(physical_log).all():
                raise FloatingPointError("Estimator produced nonfinite log parameters")
            with np.errstate(over="raise", invalid="raise"):
                target = np.exp(physical_log)
        else:
            target = norm["prior"]
        if not np.isfinite(target).all() or np.any(np.asarray(target) <= 0):
            raise FloatingPointError("Estimator produced invalid physical parameters")
        out_of_range = (np.asarray(target) < norm.get("target_min", [-np.inf, -np.inf])) | (
                       np.asarray(target) > norm.get("target_max", [np.inf, np.inf]))
        return dict(mass_kg=float(target[0]), friction=float(target[1]), attempts=self.attempts,
                    informative_attempts=self.informative_attempts, snippets=self.snippets,
                    out_of_training_range=dict(mass_kg=bool(out_of_range[0]), friction=bool(out_of_range[1])))
