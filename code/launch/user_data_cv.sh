#!/bin/bash
# user_data_cv.sh — benchmark-resolution killing experiment, one shard per g4dn.xlarge (T4).
# Template: __SHARD__ / __NSHARDS__ substituted by launch_cv.sh. Mirrors the JOSA A October runner
# (code/user_data.sh) with the lessons recorded in HANDOFF_josaa.md:
#   * /opt/pytorch/bin/activate fails on this AMI -> call /opt/pytorch/bin/python3 directly
#   * set -e + EXIT trap so a failed shard still uploads its log and terminates (no cost leak)
set -euo pipefail
exec > /var/log/benchres_cv.log 2>&1

S3=s3://<your-bucket>/JEI_IntrinsicUQ
S3F=$S3/followup_bench_resolution
SHARD=__SHARD__
NSHARDS=__NSHARDS__
RES=$S3F/results_cv_v1/shard$SHARD

finish() {
  aws s3 sync /home/ubuntu/results $RES --quiet || true
  aws s3 cp /var/log/benchres_cv.log $RES/run.log || true
  shutdown -h now
}
trap finish EXIT

mkdir -p /home/ubuntu/{code,data,results}
cd /home/ubuntu

# periodic log sync (visibility while running)
( while true; do sleep 300; aws s3 cp /var/log/benchres_cv.log $RES/run.log --quiet || true; done ) &

# data + code (same Sintel tarball as the JOSA A grid)
aws s3 cp $S3/data/sintel_training.tar /tmp/d.tar
tar xf /tmp/d.tar -C /home/ubuntu/data && rm /tmp/d.tar
aws s3 cp $S3F/code/experiment_cv.py code/
aws s3 cp $S3F/code/v7_base.py code/

PY=/opt/pytorch/bin/python3
$PY -c "import torch, torchvision, PIL, numpy; print(torch.__version__, torch.cuda.is_available())" \
  || $PY -m pip install -q pillow
nvidia-smi -L || true

cd /home/ubuntu/code
$PY experiment_cv.py \
  --data-root /home/ubuntu/data/training \
  --out-dir /home/ubuntu/results \
  --archs direct,unet --reps 5 --folds 5 --seeds 42,43 \
  --epochs 60 --batch-size 16 --size 256 --num-workers 4 \
  --shard $SHARD --num-shards $NSHARDS \
  --s3-prefix $RES
# EXIT trap: final sync + log + shutdown(terminate)
