#!/bin/bash

for ((i=0;i<3;i+=1))
do
    python3 main.py \
        --env_id="Humanoid-v5" \
        --seed=$i \
        --action_scale=0.4

    python3 main.py \
        --env_id="HalfCheetah-v5" \
        --seed=$i

    python3 main.py \
        --env_id="Ant-v5" \
        --seed=$i

    python3 main.py \
        --env_id="Walker2d-v5" \
        --seed=$i

    python3 main.py \
        --env_id="Hopper-v5" \
        --seed=$i
done