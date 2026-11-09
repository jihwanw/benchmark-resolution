#!/bin/bash
# user_data_flow_prep.sh — one-off: download Sintel flow extras, verify, stage to S3, calibrate RAFT run times on a T4.
set -euo pipefail
exec > /var/log/flow_prep.log 2>&1
S3=s3://<your-bucket>/JEI_IntrinsicUQ
S3F=$S3/followup_bench_resolution
RES=$S3F/results_flow_v1/prep
finish() { aws s3 cp /var/log/flow_prep.log $RES/prep.log || true; shutdown -h now; }
trap finish EXIT
( while true; do sleep 180; aws s3 cp /var/log/flow_prep.log $RES/prep.log --quiet || true; done ) &

mkdir -p /home/ubuntu/{code,data,results_flow} && cd /home/ubuntu
# clean images (same tar as the intrinsic arm)
aws s3 cp $S3/data/sintel_training.tar /tmp/d.tar && tar xf /tmp/d.tar -C /home/ubuntu/data && rm /tmp/d.tar
ls /home/ubuntu/data/training | head
# flow ground truth + masks (official European mirror on sintel.is.tue.mpg.de/downloads; the UW mirror host no longer resolves, 2026-11-03)
cd /tmp && curl -sSL --retry 3 -o extras.zip "https://files.is.tue.mpg.de/sintel/MPI-Sintel-training_extras.zip"
ls -la extras.zip; unzip -t extras.zip > /tmp/unzip_test.txt 2>&1; tail -1 /tmp/unzip_test.txt
unzip -q extras.zip -d /tmp/extras
find /tmp/extras -maxdepth 2 | head -20
mv /tmp/extras/training/flow /home/ubuntu/data/training/flow
mv /tmp/extras/training/invalid /home/ubuntu/data/training/invalid
mv /tmp/extras/training/occlusions /home/ubuntu/data/training/occlusions
# keep the reader code for provenance; print the header convention from the official C reader
mkdir -p /home/ubuntu/data/flow_code && cp -r /tmp/extras/flow_code/* /home/ubuntu/data/flow_code/ 2>/dev/null || true
grep -n -i -E "TAG_FLOAT|PIEH|202021" /home/ubuntu/data/flow_code/C/flowIO.cpp /home/ubuntu/data/flow_code/C/flowIO.h 2>/dev/null | head -5 || true
echo "flo files: $(find /home/ubuntu/data/training/flow -name '*.flo' | wc -l), invalid: $(find /home/ubuntu/data/training/invalid -name '*.png' | wc -l), occ: $(find /home/ubuntu/data/training/occlusions -name '*.png' | wc -l)"
# stage extras for the grid instances
cd /home/ubuntu/data/training && tar cf /tmp/sintel_flow_extras.tar flow invalid occlusions && aws s3 cp /tmp/sintel_flow_extras.tar $S3/data/sintel_flow_extras.tar --quiet && rm /tmp/sintel_flow_extras.tar
aws s3 cp /home/ubuntu/data/flow_code $S3/data/sintel_flow_code --recursive --quiet || true
rm -rf /tmp/extras /tmp/extras.zip

cd /home/ubuntu/code && aws s3 cp $S3F/code_flow/experiment_flow.py .
PY=/opt/pytorch/bin/python3
$PY -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
nvidia-smi -L || true
# header check with the exact reader used by the experiment
$PY - <<'EOF'
import glob, numpy as np, sys
sys.path.insert(0, '/home/ubuntu/code'); from experiment_flow import read_flo, list_pairs
f = sorted(glob.glob('/home/ubuntu/data/training/flow/*/*.flo'))[0]
fl = read_flo(f); print('flo check', f, fl.shape, fl.dtype, 'mean|flow|', float(np.linalg.norm(fl, axis=2).mean()))
pairs = list_pairs('/home/ubuntu/data/training'); print('pairs', len(pairs), 'scenes', len({p[0] for p in pairs}))
EOF
# calibration: 60 iterations each at the pre-registered budget settings
$PY experiment_flow.py --data-root /home/ubuntu/data/training --out-dir /home/ubuntu/results_flow --archs raft_small,raft_large \
  --calibrate 60 --iters 3000 --batch-size 4 --crop 320,448 --num-workers 4 --eval-batch 4 --s3-prefix $RES
echo PREP_DONE
