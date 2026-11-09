#!/bin/bash
# launch_cv2.sh — upload v2 code to S3 and launch N g4dn.xlarge shards (us-east-1). Usage: bash launch_cv2.sh 16
set -euo pipefail
N=${1:-16}
HERE=$(cd "$(dirname "$0")" && pwd); CODE=$(cd "$HERE/.." && pwd)
S3F=s3://<your-bucket>/JEI_IntrinsicUQ/followup_bench_resolution
REGION=us-east-1; AMI=ami-09343c2dd0ee54c92; TYPE=g4dn.xlarge
for f in experiment_cv2.py experiment_v8.py v7_base.py strata_v2.json; do aws s3 cp "$CODE/$f" $S3F/code_v2/$f --quiet; done
echo "code uploaded to $S3F/code_v2/"
OUT="$HERE/instances_v2_$(date +%Y%m%d_%H%M).txt"; : > "$OUT"
for ((i=0; i<N; i++)); do
  sed -e "s/__SHARD__/$i/" -e "s/__NSHARDS__/$N/" "$HERE/user_data_cv2.sh" > /tmp/ud_cv2_$i.sh
  ID=$(aws ec2 run-instances --region $REGION --image-id $AMI --instance-type $TYPE --count 1 \
        --iam-instance-profile Name=EC2-SSM-Role --instance-initiated-shutdown-behavior terminate \
        --user-data file:///tmp/ud_cv2_$i.sh \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=BenchRes-cv-v2-shard$i},{Key=Project,Value=benchmark_resolution_v2}]" \
        --query 'Instances[0].InstanceId' --output text)
  echo "shard$i $ID" | tee -a "$OUT"
done
echo "launched $N shards; ids in $OUT"
