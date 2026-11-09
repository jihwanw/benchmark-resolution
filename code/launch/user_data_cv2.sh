#!/bin/bash
# user_data_cv2.sh — benchmark-resolution MAIN experiment (v2), one shard per g4dn.xlarge (T4).
# __SHARD__ / __NSHARDS__ substituted by launch_cv2.sh. Resume-capable: syncs the shard's S3 prefix down first.
set -euo pipefail
exec > /var/log/benchres_cv2.log 2>&1

S3=s3://<your-bucket>/JEI_IntrinsicUQ
S3F=$S3/followup_bench_resolution
SHARD=__SHARD__
NSHARDS=__NSHARDS__
RES=$S3F/results_cv_v2/shard$SHARD

finish() {
  aws s3 sync /home/ubuntu/results $RES --quiet || true
  aws s3 cp /var/log/benchres_cv2.log $RES/run.log || true
  shutdown -h now
}
trap finish EXIT

mkdir -p /home/ubuntu/{code,data,results}
cd /home/ubuntu
( while true; do sleep 300; aws s3 cp /var/log/benchres_cv2.log $RES/run.log --quiet || true; done ) &

aws s3 cp $S3/data/sintel_training.tar /tmp/d.tar
tar xf /tmp/d.tar -C /home/ubuntu/data && rm /tmp/d.tar
for f in experiment_cv2.py experiment_v8.py v7_base.py strata_v2.json; do aws s3 cp $S3F/code_v2/$f code/; done
aws s3 sync $RES /home/ubuntu/results --exclude "run.log" --quiet || true     # resume

PY=/opt/pytorch/bin/python3
$PY -c "import torch, torchvision, PIL, numpy; print(torch.__version__, torch.cuda.is_available())" || $PY -m pip install -q pillow
nvidia-smi -L || true

cd /home/ubuntu/code
$PY experiment_cv2.py \
  --data-root /home/ubuntu/data/training --out-dir /home/ubuntu/results --strata strata_v2.json \
  --archs direct,unet,transformer,proposed --reps 5 --folds 5 --seeds 42,43,44 \
  --strat-archs direct,unet --strat-reps 2 --strat-seeds 42,43 \
  --epochs 60 --batch-size 16 --size 256 --num-workers 4 \
  --shard $SHARD --num-shards $NSHARDS --s3-prefix $RES
