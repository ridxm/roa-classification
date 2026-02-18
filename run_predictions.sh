#!/bin/bash
source /common/users/rm1838/miniforge3/etc/profile.d/conda.sh && conda activate adaptive_roa
cd /common/users/rm1838/roa-classification

# Q2D runs
python -m roa_classification.predict system=quadrotor2d checkpoint=outputs/q2d_3000/2026-02-14_22-52-58/checkpoints/epoch06-val_loss0.0851.ckpt
python -m roa_classification.predict system=quadrotor2d checkpoint=outputs/q2d_4000/2026-02-14_22-52-58/checkpoints/epoch01-val_loss0.0643.ckpt
python -m roa_classification.predict system=quadrotor2d checkpoint=outputs/q2d_5000/2026-02-14_22-52-58/checkpoints/epoch06-val_loss0.0838.ckpt
python -m roa_classification.predict system=quadrotor2d checkpoint=outputs/q2d_6000/2026-02-14_22-52-58/checkpoints/epoch02-val_loss0.0677.ckpt
python -m roa_classification.predict system=quadrotor2d checkpoint=outputs/q2d_7000/2026-02-14_22-52-58/checkpoints/epoch04-val_loss0.0583.ckpt
python -m roa_classification.predict system=quadrotor2d checkpoint=outputs/q2d_8000/2026-02-14_22-52-58/checkpoints/epoch00-val_loss0.1388.ckpt
python -m roa_classification.predict system=quadrotor2d checkpoint=outputs/q2d_9000/2026-02-14_22-52-58/checkpoints/epoch06-val_loss0.0441.ckpt
python -m roa_classification.predict system=quadrotor2d checkpoint=outputs/q2d_10000/2026-02-14_22-52-58/checkpoints/epoch03-val_loss0.1006.ckpt
python -m roa_classification.predict system=quadrotor2d checkpoint=outputs/q2d_11000/2026-02-14_22-52-58/checkpoints/epoch02-val_loss0.0470.ckpt
python -m roa_classification.predict system=quadrotor2d checkpoint=outputs/q2d_12000/2026-02-14_22-52-58/checkpoints/epoch02-val_loss0.0860.ckpt

# Q3D runs
python -m roa_classification.predict system=quadrotor3d checkpoint=outputs/q3d_3000/2026-02-15_13-24-18/checkpoints/epoch00-val_loss0.1227.ckpt
python -m roa_classification.predict system=quadrotor3d checkpoint=outputs/q3d_4000/2026-02-15_13-24-16/checkpoints/epoch00-val_loss0.0713.ckpt
python -m roa_classification.predict system=quadrotor3d checkpoint=outputs/q3d_5000/2026-02-15_13-24-21/checkpoints/epoch00-val_loss0.1257.ckpt
python -m roa_classification.predict system=quadrotor3d checkpoint=outputs/q3d_6000/2026-02-15_13-25-12/checkpoints/epoch00-val_loss0.0961.ckpt
python -m roa_classification.predict system=quadrotor3d checkpoint=outputs/q3d_7000/2026-02-15_13-24-16/checkpoints/epoch00-val_loss0.1159.ckpt
python -m roa_classification.predict system=quadrotor3d checkpoint=outputs/q3d_8000/2026-02-15_13-24-21/checkpoints/epoch00-val_loss0.1151.ckpt
python -m roa_classification.predict system=quadrotor3d checkpoint=outputs/q3d_9000/2026-02-15_13-24-21/checkpoints/epoch00-val_loss0.1206.ckpt
python -m roa_classification.predict system=quadrotor3d checkpoint=outputs/q3d_10000/2026-02-15_13-24-21/checkpoints/epoch00-val_loss0.1073.ckpt
python -m roa_classification.predict system=quadrotor3d checkpoint=outputs/q3d_11000/2026-02-15_13-24-21/checkpoints/epoch00-val_loss0.0917.ckpt
python -m roa_classification.predict system=quadrotor3d checkpoint=outputs/q3d_12000/2026-02-15_13-24-21/checkpoints/epoch00-val_loss0.1155.ckpt

echo "All predictions complete!"
