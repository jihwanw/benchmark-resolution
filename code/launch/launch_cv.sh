#!/bin/bash
# launch_cv.sh — upload code to S3 and launch N g4dn.xlarge shards in us-east-1.
# Usage: bash launch_cv.sh <NSHARDS>
set -euo pipefail
N=${1:-12}
HERE=$(cd "$(dirname "$0")" && pwd)
CODE=$(cd "$HERE/.." && pwd)
S3F=s3://<your-bucket>/JEI_IntrinsicUQ/followup_bench_resolution
REGION=us-east-1
AMI=ami-09343c2dd0ee54c92          # Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.11 (Ubuntu 24.04) — same as JOSA A grid
TYPE=g4dn.xlarge                   # T4, same hardware as the 325-run grid

aws s3 cp "$CODE/experiment_cv.py" $S3F/code/experiment_cv.py
aws s3 cp "$CODE/v7_base.py" $S3F/code/v7_base.py

OUT="$HERE/instances_$(date +%Y%m%d_%H%M).txt"
: > "$OUT"
for ((i=0; i<N; i++)); do
  sed -e "s/__SHARD__/$i/" -e "s/__NSHARDS__/$N/" "$HERE/user_data_cv.sh" > /tmp/ud_cv_$i.sh
  ID=$(aws ec2 run-instances --region $REGION \
        --image-id $AMI --instance-type $TYPE --count 1 \
        --iam-instance-profile Name=EC2-SSM-Role \
        --instance-initiated-shutdown-behavior terminate \
        --user-data file:///tmp/ud_cv_$i.sh \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=BenchRes-cv-v1-shard$i},{Key=Project,Value=benchmark_resolution}]" \
        --query 'Instances[0].InstanceId' --output text)
  echo "shard$i $ID" | tee -a "$OUT"
done
echo "launched $N shards; ids in $OUT"
