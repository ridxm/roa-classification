#!/bin/bash
set -e
cd /common/users/rm1838/roa-classification

# Q3D all_shuffled
for n in 10000 17500 25000; do
  dir=$(ls -dt outputs/q3d_all_${n}/2026-* | head -1)
  ckpt=$(grep "Best checkpoint" outputs/q3d_all_${n}.log | grep -oP "/common.*\.ckpt")
  ckpt_rel=${ckpt#/common/users/rm1838/roa-classification/}
  echo "Running q3d_all_${n}..."
  python -m roa_classification.predict system=quadrotor3d checkpoint="$ckpt_rel" output_dir="$dir"
  echo "Done q3d_all_${n}"
done

# CartPole
for n in 300 700 1000; do
  dir=$(ls -dt outputs/cp_${n}/2026-02-22* | head -1)
  best=$(grep "Best checkpoint" outputs/cp_${n}.log 2>/dev/null | grep -oP "/common.*\.ckpt")
  if [ -z "$best" ]; then
    best=$(ls "$dir/checkpoints/"epoch*.ckpt | grep -v last | sort -t'l' -k2 | head -1)
  fi
  ckpt_rel=${best#/common/users/rm1838/roa-classification/}
  echo "Running cp_${n}..."
  python -m roa_classification.predict system=cartpole checkpoint="$ckpt_rel" output_dir="$dir"
  echo "Done cp_${n}"
done

# Pendulum
for n in 50 250 500; do
  dir=$(ls -dt outputs/pend_${n}/2026-02-22* | head -1)
  best=$(grep "Best checkpoint" outputs/pend_${n}.log 2>/dev/null | grep -oP "/common.*\.ckpt")
  if [ -z "$best" ]; then
    best=$(ls "$dir/checkpoints/"epoch*.ckpt | grep -v last | sort -t'l' -k2 | head -1)
  fi
  ckpt_rel=${best#/common/users/rm1838/roa-classification/}
  echo "Running pend_${n}..."
  python -m roa_classification.predict system=pendulum checkpoint="$ckpt_rel" output_dir="$dir"
  echo "Done pend_${n}"
done

echo "All predictions complete!"
