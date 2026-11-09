#!/bin/bash
# user_data_qual.sh — qualitative-figure runs. __JOB__ = intrinsic | flow. Uploads npz + meta to S3 and terminates.
set -euo pipefail
exec > /var/log/benchres_qual.log 2>&1
S3=s3://<your-bucket>/JEI_IntrinsicUQ; S3F=$S3/followup_bench_resolution; JOB=__JOB__; RES=$S3F/qual
finish() { aws s3 cp /var/log/benchres_qual.log $RES/run_$JOB.log || true; shutdown -h now; }
trap finish EXIT
( while true; do sleep 300; aws s3 cp /var/log/benchres_qual.log $RES/run_$JOB.log --quiet || true; done ) &
mkdir -p /home/ubuntu/{code,data,results_qual} && cd /home/ubuntu
aws s3 cp $S3/data/sintel_training.tar /tmp/d.tar && tar xf /tmp/d.tar -C /home/ubuntu/data && rm /tmp/d.tar
if [ "$JOB" = "flow" ]; then aws s3 cp $S3/data/sintel_flow_extras.tar /tmp/e.tar && tar xf /tmp/e.tar -C /home/ubuntu/data/training && rm /tmp/e.tar; fi
for f in experiment_cv2.py experiment_v8.py v7_base.py strata_v2.json; do aws s3 cp $S3F/code_v2/$f code/; done
aws s3 cp $S3F/code_flow/experiment_flow.py code/; aws s3 cp $S3F/code_qual/qual_intrinsic.py code/; aws s3 cp $S3F/code_qual/qual_flow.py code/
PY=/opt/pytorch/bin/python3; $PY -c "import torch; print(torch.__version__, torch.cuda.is_available())"; nvidia-smi -L || true
cd /home/ubuntu/code
if [ "$JOB" = "intrinsic" ]; then
  $PY qual_intrinsic.py --data-root /home/ubuntu/data/training --out-dir /home/ubuntu/results_qual --epochs 60 --size 256 --batch-size 16 --num-workers 4 --s3-prefix $RES
else
  $PY qual_flow.py --data-root /home/ubuntu/data/training --out-dir /home/ubuntu/results_qual --iters 3000 --batch-size 4 --crop 320,448 --num-workers 4 --s3-prefix $RES
fi
