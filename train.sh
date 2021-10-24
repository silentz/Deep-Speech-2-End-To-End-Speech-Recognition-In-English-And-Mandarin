#!/bin/bash

python trainer.py fit \
    --config config/stage1.yaml

latest_file=$(ls -t1 ./checkpoints/stage1/ | head -n 1)

python trainer.py fit \
    --config config/stage2.yaml \
    --trainer.resume_from_checkpoint "./checkpoints/stage1/$latest_file"
