#!/bin/bash
# user_data_flow.sh — Sintel optical-flow arm (v1), one shard per g4dn.xlarge (T4). Resume-capable.
# __SHARD__ / __NSHARDS__ / __EXPECTED__ substituted by launch_flow.sh (EXPECTED = raft_small=MM,raft_large=MM from calibration).
set -euo pipefail
exec > /var/log/benchres_flow.log 2>&1
S3=s3://<your-bucket>/JEI_IntrinsicUQ
S3F=$S3/followup_bench_resolution
SHARD=__SHARD__; NSHARDS=__NSHARDS__; EXPECTED="__EXPECTED__"
RES=$S3F/results_flow_v1/shard$SHARD
finish() { aws s3 sync /home/ubuntu/results_flow $RES --quiet || true; aws s3 cp /var/log/benchres_flow.log $RES/run.log || true; shutdown -h now; }
trap finish EXIT
( while true; do sleep 300; aws s3 cp /var/log/benchres_flow.log $RES/run.log --quiet || true; done ) &

mkdir -p /home/ubuntu/{code,data,results_flow} && cd /home/ubuntu
aws s3 cp $S3/data/sintel_training.tar /tmp/d.tar && tar xf /tmp/d.tar -C /home/ubuntu/data && rm /tmp/d.tar
aws s3 cp $S3/data/sintel_flow_extras.tar /tmp/e.tar && tar xf /tmp/e.tar -C /home/ubuntu/data/training && rm /tmp/e.tar
aws s3 cp $S3F/code_flow/experiment_flow.py code/
aws s3 sync $RES /home/ubuntu/results_flow --exclude "run.log" --quiet || true    # resume
PY=/opt/pytorch/bin/python3
$PY -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
nvidia-smi -L || true
cd /home/ubuntu/code
$PY experiment_flow.py --data-root /home/ubuntu/data/training --out-dir /home/ubuntu/results_flow \
  --archs raft_small,raft_large --reps 5 --folds 5 --seeds 42,43,44 \
  --iters 3000 --batch-size 4 --crop 320,448 --num-workers 4 --eval-batch 4 \
  --shard $SHARD --num-shards $NSHARDS --expected-minutes "$EXPECTED" --s3-prefix $RES
