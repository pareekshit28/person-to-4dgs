#!/bin/bash
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=1 python src/simple_trainer_freetime_4d_pure_relocation.py default_keyframe_small \
    --data-dir /data/shared/elaheh/4D_demo/completed_indoor/mike_tech/undistorted \
    --init-npz-path /data/shared/elaheh/4D_demo/completed_indoor/mike_tech/undistorted/keyframes_smart_15M_0to60.npz \
    --result-dir /data/shared/elaheh/mike_test_free_gs_60frames_small \
    --start-frame 0 \
    --end-frame 61 \
    --max-steps 50000 \
    --eval-steps 50000 \
    --save-steps 50000
