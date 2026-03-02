#!/bin/bash
# Run predict_splits for all trained models.
# Finds the best checkpoint and generates train/val/test CSVs.
set -e

PROJ=/common/users/rm1838/roa-classification
cd $PROJ

predict() {
  local system=$1
  local name=$2
  local num_traj=$3
  local extra_args=$4

  # Find the latest output dir for this run
  dir=$(ls -dt outputs/${name}/2026-* 2>/dev/null | head -1)
  if [ -z "$dir" ]; then
    echo "SKIP $name: no output dir found"
    return
  fi

  # Find best checkpoint (lowest val_loss, not last.ckpt)
  best=$(for f in "$dir/checkpoints/"epoch*.ckpt; do v=$(echo "$f" | grep -oP 'val_loss\K[\d.]+'); echo "$v $f"; done | sort -n | head -1 | awk '{print $2}')
  if [ -z "$best" ]; then
    echo "SKIP $name: no checkpoint found in $dir/checkpoints/"
    return
  fi

  echo "=== $name ==="
  echo "  Dir:  $dir"
  echo "  Ckpt: $best"
  python -m roa_classification.predict_splits \
    system=$system \
    checkpoint="$best" \
    output_dir="$dir" \
    data.num_trajectories=$num_traj \
    $extra_args
  echo ""
}

# Q3D (all_shuffled)
predict quadrotor3d q3d_split_10000 10000 "data.split_name=all_shuffled"
predict quadrotor3d q3d_split_17000 17000 "data.split_name=all_shuffled"
predict quadrotor3d q3d_split_25000 25000 "data.split_name=all_shuffled"

# Q2D
predict quadrotor2d q2d_split_3000  3000
predict quadrotor2d q2d_split_7000  7000
predict quadrotor2d q2d_split_12000 12000

# Pendulum
predict pendulum pend_split_50  50
predict pendulum pend_split_250 250
predict pendulum pend_split_500 500

# CartPole
predict cartpole cp_split_300  300
predict cartpole cp_split_700  700
predict cartpole cp_split_1000 1000

echo "All predictions complete!"
