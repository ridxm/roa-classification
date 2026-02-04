# ROA Classification

Binary classifier for Region of Attraction (ROA) estimation of dynamical systems. Given a state, the model predicts whether the system will reach the goal (class 1) or fail (class 0).

Built with **PyTorch Lightning** and **Hydra** for configuration management.

## Supported Systems

| System | Raw State | Embedded State | Manifold |
|--------|-----------|----------------|----------|
| CartPole PyBullet | `[x, theta, x_dot, theta_dot]` (4D) | `[x_norm, sin(theta), cos(theta), x_dot_norm, theta_dot_norm]` (5D) | R x S1 x R^2 |
| Pendulum | `[theta, theta_dot]` (2D) | `[sin(theta), cos(theta), theta_dot_norm]` (3D) | S1 x R |

Circular state components (angles) are embedded as `(sin, cos)` pairs to respect the manifold topology. Real-valued components are normalized by symmetric bounds from `dataset_description.json`.

## Setup

```bash
conda activate adaptive_roa
cd /path/to/roa-classification
pip install -e .
```

## Usage

### Training

```bash
# default: cartpole, mlp_small, 500 epochs, gpu0
python -m roa_classification.train

# specify system and model
python -m roa_classification.train system=pendulum model=mlp_large

# override training params
python -m roa_classification.train trainer.max_epochs=1000 data.batch_size=512

# select GPU
python -m roa_classification.train device=gpu3
```

### Evaluation

```bash
python -m roa_classification.evaluate checkpoint=/path/to/e042.ckpt
python -m roa_classification.evaluate system=pendulum checkpoint=/path/to/e100.ckpt
```

### Common Overrides

```bash
system=cartpole|pendulum         # dynamical system
model=mlp_small|mlp_large        # model architecture
device=cpu|gpu0|gpu1|...|gpu5    # compute device
trainer.max_epochs=1000          # training epochs
data.num_trajectories=500        # number of trajectories to load
data.balance_samples=false       # disable class balancing
model.lower_thresh=0.3           # lower separatrix threshold
model.upper_thresh=0.7           # upper separatrix threshold
```

## Data Flow

### Training and Validation

1. Load first `N` trajectory filenames from `train_test_splits/shuffled_indices_0.txt` (default N=1000)
2. Read matching labels from `train_test_splits/shuffled_labels_0.txt`
3. For each trajectory file, load **all states** (every line) — each state inherits the trajectory's 0/1 label
4. Embed each raw state using the system's manifold-aware embedding
5. Split all states into **95% train / 5% val**
6. Train split uses `WeightedRandomSampler` for class balancing (inverse frequency weighting)

### Test (post-training)

After training completes, the model is evaluated on `eval_states.txt`:
- Each row contains initial state + final state + label
- Only the initial state (first 4 columns) is used as input
- Label (column 9) is the ground truth
- This runs once via `trainer.test()`, not every epoch

### Data Format

```
trajectories/              # per-trajectory state files (variable length)
  sequence_00001.txt       # each line: x,theta,x_dot,theta_dot
  sequence_00002.txt
  ...
train_test_splits/
  shuffled_indices_0.txt   # trajectory filenames (shuffled)
  shuffled_labels_0.txt    # aligned 0/1 labels
eval_states.txt            # x,theta,x_dot,theta_dot,x_f,theta_f,x_dot_f,theta_dot_f,label
dataset_description.json   # metadata including achieved_bounds for normalization
```

## Model

**ClassifierMLP** — feedforward MLP with configurable hidden layers.

| Config | Hidden Layers | Parameters |
|--------|---------------|------------|
| `mlp_small` | [128, 256, 128] | ~67K |
| `mlp_large` | [512, 1024, 512] | ~1.1M |

- **Loss**: BCEWithLogitsLoss
- **Optimizer**: AdamW (lr=1e-3, weight_decay=1e-4)
- **Scheduler**: CosineAnnealingLR (T_max = max_epochs, eta_min=1e-6)
- **Callbacks**: ModelCheckpoint (top-3 by val/loss), EarlyStopping (patience=50)

### Threshold-Based Evaluation

Instead of a hard 0.5 decision boundary, the model uses a **separatrix** region:

| Probability | Prediction |
|-------------|------------|
| p < 0.4 | Class 0 (failure) |
| 0.4 <= p <= 0.6 | Separatrix (unclassified) |
| p > 0.6 | Class 1 (success) |

Metrics are computed on **classified samples only**: Precision, Recall, Specificity, F1, and Separatrix %.

## Project Structure

```
roa-classification/
  configs/
    train.yaml              # main training config
    evaluate.yaml           # evaluation config
    system/                 # system-specific configs
      cartpole.yaml
      pendulum.yaml
    model/                  # model architecture configs
      mlp_small.yaml
      mlp_large.yaml
    device/                 # compute device configs
      cpu.yaml
      gpu0.yaml ... gpu5.yaml
  roa_classification/
    train.py                # training entry point
    evaluate.py             # evaluation entry point
    data/
      base.py               # BaseClassificationDataModule (train/val/test loaders)
      cartpole.py            # CartPole data module
      pendulum.py            # Pendulum data module
    model/
      mlp.py                 # ClassifierMLP (Lightning module)
    systems/
      base.py                # DynamicalSystem ABC + ManifoldComponent
      cartpole.py            # CartPole system (embedding, bounds)
      pendulum.py            # Pendulum system (embedding, bounds)
    utils/
      env_config.py          # .env file loader
  outputs/                   # training outputs (gitignored)
```

## Example Output

```
================================================================================
Cartpole Classification Training
================================================================================
Config: cartpole
Seed: 42
Max epochs: 500
================================================================================

Loading data...
Loaded 98686 samples from 1000 trajectories
  Success (1): 88835 (94.7%)
  Failure (0): 4917 (5.3%)
Train: 93751 samples (95%)
Val:   4935 samples (5%)
Test:  116242 samples (eval_states.txt)

Creating model...
Model: ClassifierMLP
Input dim: 5
Hidden layers: [128, 256, 128]
Parameters: 66,817

Starting training...
================================================================================

[val] thresholds=(0.4, 0.6)  total=4935  classified=4812
    Separatrix   Precision      Recall   Specificity          F1
         2.49%      0.9934      0.9978        0.9212      0.9956

================================================================================
Training complete!
Best checkpoint: outputs/cartpole/2026-02-04_14-00-00/lightning_logs/.../e042.ckpt
================================================================================
```
