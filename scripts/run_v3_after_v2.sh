#!/usr/bin/env bash
set -euo pipefail

workspace=/home/dzliu/pzliu/sap-rl
image=nvcr.io/nvidia/pytorch:25.09-py3
v2_container=sap-rl-turtle-league-v2
v3_container=sap-rl-turtle-league-v3
dataset=/workspace/replay-data/expert/guide-bootstrap-v3.jsonl
initialization=/workspace/checkpoints/imitation-guide-v3.pt
run_dir=/workspace/runs/turtle-league-v3-audited

if docker inspect "$v2_container" >/dev/null 2>&1; then
    if [ "$(docker inspect -f '{{.State.Running}}' "$v2_container")" = true ]; then
        echo "waiting_for=$v2_container"
        docker wait "$v2_container"
    fi
fi

echo "v2_finished_at=$(date --iso-8601=seconds)"

if [ ! -f "$workspace/checkpoints/imitation-guide-v3.pt" ]; then
    echo "starting_v3_pretraining"
    docker run --rm --gpus all --ipc=host \
        -v "$workspace:/workspace" \
        -w /workspace/sapai-gym \
        -e PYTHONPATH=/workspace/sapai:/workspace/sapai-gym \
        "$image" bash -lc \
        "python -m pip install -q gymnasium==0.29.1 stable_baselines3==2.7.0 sb3-contrib==2.3.0 scikit-learn graphviz && exec python scripts/pretrain_imitation.py '$dataset' --epochs 12 --batch-size 64 --learning-rate 0.0003 --output '$initialization' --device cuda --seed 20260829"
fi

mkdir -p "$workspace/runs/turtle-league-v3-audited"

if docker inspect "$v3_container" >/dev/null 2>&1; then
    echo "v3_container_already_exists=$v3_container"
    exit 0
fi

echo "starting_v3_league"
docker run -d --gpus all --ipc=host \
    --name "$v3_container" \
    -v "$workspace:/workspace" \
    -w /workspace/sapai-gym \
    -e PYTHONPATH=/workspace/sapai:/workspace/sapai-gym \
    "$image" bash -lc \
    "python -m pip install -q gymnasium==0.29.1 stable_baselines3==2.7.0 sb3-contrib==2.3.0 scikit-learn graphviz && exec python scripts/train_entity_selfplay.py --distributed --init-checkpoint '$initialization' --demo-data '$dataset' --demo-batch-size 64 --demo-until-update 250 --demo-every 4 --demo-learning-rate-scale 0.2 --updates 1250 --num-envs 16 --games-per-actor 1 --segments-per-update 64 --transitions-per-update 8192 --checkpoint-every 50 --checkpoint-dir '$run_dir'"

echo "v3_container_started=$v3_container"

