#!/bin/bash
# launch_flow.sh — launch the flow-arm grid after calibration. Usage: bash launch_flow.sh <NSHARDS> "raft_small=MM,raft_large=MM"
set -euo pipefail
N=${1:-16}; EXPECTED=${2:?expected minutes, e.g. raft_small=22,raft_large=55}
HERE=$(cd "$(dirname "$0")" && pwd); CODE=$(cd "$HERE/.." && pwd)
S3F=s3://<your-bucket>/JEI_IntrinsicUQ/followup_bench_resolution
REGION=us-east-1; AMI=ami-09343c2dd0ee54c92; TYPE=g4dn.xlarge
aws s3 cp "$CODE/experiment_flow.py" $S3F/code_flow/experiment_flow.py --quiet
aws s3 ls s3://<your-bucket>/JEI_IntrinsicUQ/data/sintel_flow_extras.tar >/dev/null || { echo "extras tar missing in S3 — run prep first"; exit 1; }
OUT="$HERE/instances_flow_$(date +%Y%m%d_%H%M).txt"; : > "$OUT"
for ((i=0; i<N; i++)); do
  sed -e "s/__SHARD__/$i/" -e "s/__NSHARDS__/$N/" -e "s/__EXPECTED__/$EXPECTED/" "$HERE/user_data_flow.sh" > /tmp/ud_flow_$i.sh
  ID=$(aws ec2 run-instances --region $REGION --image-id $AMI --instance-type $TYPE --count 1 \
        --iam-instance-profile Name=EC2-SSM-Role --instance-initiated-shutdown-behavior terminate \
        --user-data file:///tmp/ud_flow_$i.sh \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=BenchRes-flow-v1-shard$i},{Key=Project,Value=benchmark_resolution_flow}]" \
        --query 'Instances[0].InstanceId' --output text)
  echo "shard$i $ID" | tee -a "$OUT"
done
echo "launched $N shards; ids in $OUT"
