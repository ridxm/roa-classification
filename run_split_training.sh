#!/bin/bash
# Launch all trajectory-level split training runs in separate tmux windows.
# Each run gets its own GPU via srun.
set -e

PROJ=/common/users/rm1838/roa-classification
CONDA="source /common/users/rm1838/miniforge3/etc/profile.d/conda.sh && conda activate adaptive_roa"
SRUN="srun --gres=gpu:1 --mem=16G --cpus-per-task=8 --pty bash -c"

SESSION="split_train"
tmux new-session -d -s $SESSION -n dummy 2>/dev/null || true

launch() {
  local name=$1
  local cmd=$2
  echo "Launching $name..."
  tmux new-window -t $SESSION -n "$name"
  tmux send-keys -t $SESSION:"$name" "$SRUN '$CONDA && cd $PROJ && $cmd'" Enter
}

# Q3D (all_shuffled)
launch q3d_10k "python -m roa_classification.train system=quadrotor3d data.num_trajectories=10000 data.split_name=all_shuffled name=q3d_split_10000"
launch q3d_17k "python -m roa_classification.train system=quadrotor3d data.num_trajectories=17000 data.split_name=all_shuffled name=q3d_split_17000"
launch q3d_25k "python -m roa_classification.train system=quadrotor3d data.num_trajectories=25000 data.split_name=all_shuffled name=q3d_split_25000"

# Q2D
launch q2d_3k  "python -m roa_classification.train system=quadrotor2d data.num_trajectories=3000  name=q2d_split_3000"
launch q2d_7k  "python -m roa_classification.train system=quadrotor2d data.num_trajectories=7000  name=q2d_split_7000"
launch q2d_12k "python -m roa_classification.train system=quadrotor2d data.num_trajectories=12000 name=q2d_split_12000"

# Pendulum
launch pend_50  "python -m roa_classification.train system=pendulum data.num_trajectories=50  name=pend_split_50"
launch pend_250 "python -m roa_classification.train system=pendulum data.num_trajectories=250 name=pend_split_250"
launch pend_500 "python -m roa_classification.train system=pendulum data.num_trajectories=500 name=pend_split_500"

# CartPole
launch cp_300  "python -m roa_classification.train system=cartpole data.num_trajectories=300  name=cp_split_300"
launch cp_700  "python -m roa_classification.train system=cartpole data.num_trajectories=700  name=cp_split_700"
launch cp_1000 "python -m roa_classification.train system=cartpole data.num_trajectories=1000 name=cp_split_1000"

echo ""
echo "All 12 training runs launched in tmux session '$SESSION'."
echo "Monitor with: tmux attach -t $SESSION"
