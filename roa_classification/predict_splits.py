"""Generate prediction CSVs for train/val/test splits.

For train and val: reads all trajectory states, embeds, runs inference.
For test: reads test_set.txt initial states, embeds, runs inference.

Output: 3 CSVs with columns <raw_state_cols..., predicted_prob, ground_truth>.

Usage:
    python -m roa_classification.predict_splits \
        system=cartpole \
        checkpoint=/path/to/best.ckpt \
        output_dir=/path/to/training/output \
        data.num_trajectories=1000
"""

import csv
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from roa_classification.model.mlp import ClassifierMLP


def _read_trajectory_raw_states(trajectory_files, trajectory_labels, trajectories_dir):
    """Read raw states and labels from trajectory files."""
    raw_states = []
    labels = []
    for fname, label in zip(trajectory_files, trajectory_labels):
        filepath = Path(trajectories_dir) / fname
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                state = [float(v) for v in line.split(",")]
                raw_states.append(state)
                labels.append(label)
    return raw_states, labels


def _run_inference(raw_states, embed_fn, model, device, batch_size=512):
    """Embed raw states, run model inference, return probabilities."""
    embedded = np.stack(
        [embed_fn(np.array(s, dtype=np.float32)) for s in raw_states]
    )
    embedded_tensor = torch.from_numpy(embedded).float()

    all_probs = []
    with torch.no_grad():
        for i in range(0, len(embedded_tensor), batch_size):
            batch = embedded_tensor[i : i + batch_size].to(device)
            logits = model(batch)
            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().numpy().tolist())

    return all_probs


def _write_csv(output_path, header, raw_states, probs, labels):
    """Write prediction CSV."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for raw, prob, label in zip(raw_states, probs, labels):
            writer.writerow(raw + [f"{prob:.6f}", label])
    print(f"  Saved {output_path.name}: {len(raw_states)} rows")


@hydra.main(
    version_base=None, config_path="../configs", config_name="predict_splits"
)
def main(cfg: DictConfig) -> None:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Instantiate data module to get system + split
    data_module = hydra.utils.instantiate(cfg.data)
    data_module.prepare_data()
    data_module.setup()

    system = data_module.system
    embed_fn = system.embed_state
    state_dim = system.state_dim
    trajectories_dir = data_module.data_dir / "trajectories"

    # Load model
    model = ClassifierMLP.load_from_checkpoint(
        cfg.checkpoint, weights_only=False
    )
    model.eval()
    model.freeze()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Header
    state_names = [c.name for c in system.manifold_components]
    header = state_names + ["predicted_prob", "ground_truth"]

    print(f"\nGenerating predictions for {cfg.system.name}...")
    print(f"  Train trajectories: {len(data_module.train_trajectory_files)}")
    print(f"  Val trajectories:   {len(data_module.val_trajectory_files)}")
    print(f"  Output dir: {output_dir}")

    # --- Train predictions ---
    print("\nProcessing train split...")
    raw_states, labels = _read_trajectory_raw_states(
        data_module.train_trajectory_files,
        data_module.train_trajectory_labels,
        trajectories_dir,
    )
    probs = _run_inference(raw_states, embed_fn, model, device)
    _write_csv(
        output_dir / "train_predictions.csv", header, raw_states, probs, labels
    )

    # --- Val predictions ---
    print("Processing val split...")
    raw_states, labels = _read_trajectory_raw_states(
        data_module.val_trajectory_files,
        data_module.val_trajectory_labels,
        trajectories_dir,
    )
    probs = _run_inference(raw_states, embed_fn, model, device)
    _write_csv(
        output_dir / "val_predictions.csv", header, raw_states, probs, labels
    )

    # --- Test predictions ---
    print("Processing test split...")
    test_path = data_module.data_dir / data_module.test_file
    raw_states = []
    labels = []
    with open(test_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = line.split(",")
            raw_states.append([float(v) for v in vals[:state_dim]])
            labels.append(int(float(vals[-1])))
    probs = _run_inference(raw_states, embed_fn, model, device)
    _write_csv(
        output_dir / "test_predictions.csv", header, raw_states, probs, labels
    )

    print(f"\nAll predictions saved to {output_dir}")


if __name__ == "__main__":
    main()
