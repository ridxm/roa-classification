"""Generate prediction CSVs: <start_state, predicted_prob, ground_truth_label>.

Usage:
    python -m roa_classification.predict system=quadrotor2d checkpoint=/path/to/ckpt
"""

import csv
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from roa_classification.model.mlp import ClassifierMLP


@hydra.main(version_base=None, config_path="../configs", config_name="evaluate")
def main(cfg: DictConfig) -> None:
    # Instantiate data module to get system + test data
    data_module = hydra.utils.instantiate(cfg.data)
    data_module.prepare_data()
    data_module.setup(stage="test")

    system = data_module.system
    state_dim = system.state_dim

    # Load model
    model = ClassifierMLP.load_from_checkpoint(cfg.checkpoint, weights_only=False)
    model.eval()
    model.freeze()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Read raw initial states and labels from eval_states.txt
    eval_path = Path(cfg.data.dataset_dir) / "eval_states.txt"
    raw_states = []
    labels = []
    with open(eval_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = line.split(",")
            raw_states.append([float(v) for v in vals[:state_dim]])
            labels.append(int(float(vals[-1])))

    # Run inference in batches using the test dataloader
    all_probs = []
    test_loader = data_module.test_dataloader()
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["inputs"].to(device)
            logits = model(inputs)
            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().numpy().tolist())

    # Write CSV
    output_path = Path(cfg.checkpoint).parent.parent / "predictions.csv"

    # Build header
    state_names = [c.name for c in system.manifold_components]
    header = state_names + ["predicted_prob", "ground_truth"]

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for raw, prob, label in zip(raw_states, all_probs, labels):
            writer.writerow(raw + [f"{prob:.6f}", label])

    print(f"Predictions saved to {output_path}")
    print(f"  Samples: {len(raw_states)}")


if __name__ == "__main__":
    main()
