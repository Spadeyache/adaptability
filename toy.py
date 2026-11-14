"""Test whether a belief fitted on a block's lifts improves slide predictions."""

import numpy as np
import torch
from torch import nn


TRAIN_BLOCKS = 200
TEST_BLOCKS = 30
STEPS_PER_TASK = 200
Z_DIM = 4
SEED = 42


def make_blocks(rng, count):
    """Return (velocity, force, flag) and next velocity, grouped by block."""
    shape = (count, 2 * STEPS_PER_TASK)
    mass = rng.uniform(0.5, 3.0, (count, 1))
    friction = rng.uniform(0.1, 0.8, (count, 1))

    # Independent one-step probes keep initial velocity independent of block mass.
    velocity = rng.uniform(-2.0, 2.0, shape)
    force = rng.uniform(0.0, 40.0, shape)
    flag = np.broadcast_to(np.repeat([0, 1], STEPS_PER_TASK), shape)
    acceleration = force / mass - 9.8 * np.where(flag == 0, 1.0, friction)
    next_velocity = velocity + 0.05 * acceleration

    inputs = np.stack((velocity, force, flag), axis=-1).astype(np.float32)
    return torch.from_numpy(inputs), torch.from_numpy(next_velocity.astype(np.float32))


class WorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("input_scale", torch.tensor([2.0, 40.0, 1.0]))
        self.mlp = nn.Sequential(
            nn.Linear(3 + Z_DIM, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, inputs, z):
        features = torch.cat((inputs / self.input_scale, z), dim=-1)
        delta = self.mlp(features).squeeze(-1)
        return inputs[..., 0] + delta


def train_model(inputs, targets):
    model = WorldModel()
    table = nn.Embedding(TRAIN_BLOCKS, Z_DIM)
    nn.init.zeros_(table.weight)
    optimizer = torch.optim.Adam(
        [*model.parameters(), *table.parameters()], lr=0.002
    )
    loss_fn = nn.MSELoss()
    steps_per_block = inputs.shape[1]
    inputs = inputs.flatten(0, 1)
    targets = targets.flatten()

    for _ in range(60):
        for rows in torch.randperm(len(targets)).split(1024):
            block_ids = rows // steps_per_block
            predictions = model(inputs[rows], table(block_ids))
            loss = loss_fn(predictions, targets[rows])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def test_model(model, inputs, targets):
    model.eval()
    model.requires_grad_(False)
    lift_inputs, slide_inputs = inputs.split(STEPS_PER_TASK, dim=1)
    lift_targets, slide_targets = targets.split(STEPS_PER_TASK, dim=1)

    # Each row is a fresh belief. It contributes only to its own block's loss.
    z = nn.Parameter(torch.zeros(len(inputs), 1, Z_DIM))
    optimizer = torch.optim.Adam([z], lr=0.02)
    loss_fn = nn.MSELoss()
    for _ in range(150):
        predictions = model(lift_inputs, z.expand(-1, STEPS_PER_TASK, -1))
        loss = loss_fn(predictions, lift_targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        learned_z = z.expand(-1, STEPS_PER_TASK, -1)
        learned_error = loss_fn(model(slide_inputs, learned_z), slide_targets)
        zero_error = loss_fn(model(slide_inputs, torch.zeros_like(learned_z)), slide_targets)
    return learned_error.item(), zero_error.item()


def save_chart(learned_error, zero_error, filename):
    """Write an SVG bar chart using the same raw MSE values printed below."""
    bars = []
    scale = 320 / max(learned_error, zero_error, 1e-12)
    for label, error, color, y in [
        ("Lift-fitted z", learned_error, "#147d75", 82),
        ("Zero z", zero_error, "#67758a", 140),
    ]:
        width = error * scale
        bars.append(f'<text x="116" y="{y + 21}" text-anchor="end">{label}</text>')
        bars.append(f'<rect x="130" y="{y}" width="{width:.3f}" height="32" fill="{color}"/>')
        bars.append(f'<text x="{140 + width:.3f}" y="{y + 21}">{error:.6f}</text>')

    with open(filename, "w", encoding="utf-8") as chart:
        chart.write(
            '<svg xmlns="http://www.w3.org/2000/svg" width="570" height="205" '
            'viewBox="0 0 570 205" font-family="sans-serif" font-size="15" fill="#182630">\n'
            '<title>Slide prediction error on held-out blocks</title>\n'
            '<rect width="570" height="205" fill="white"/>\n'
            '<text x="18" y="32" font-size="21">Slide prediction error</text>\n'
            f'<text x="18" y="57">Next-velocity MSE across {TEST_BLOCKS} held-out blocks</text>\n'
            '<path d="M130 74 V182" stroke="#9aa6ae"/>\n'
            + "\n".join(bars) + "\n</svg>\n"
        )


def main():
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    model = train_model(*make_blocks(rng, TRAIN_BLOCKS))
    learned_error, zero_error = test_model(model, *make_blocks(rng, TEST_BLOCKS))
    print(f"Slide MSE with lift-fitted z: {learned_error:.6f}")
    print(f"Slide MSE with zero z:        {zero_error:.6f}")
    save_chart(learned_error, zero_error, "toy_slide_mse.svg")


if __name__ == "__main__":
    main()
