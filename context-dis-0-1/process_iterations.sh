#!/bin/bash

set -e

data_source=$1
data_type=$2
feature_suffix=$3
model_suffix=$4
save_format=$5
top_k=$6

echo "Data Source: $data_source"
echo "Data Type: $data_type"
echo "Feature Suffix: $feature_suffix"
echo "Model Suffix: $model_suffix"
echo "Save Format: $save_format"
echo "Top K: $top_k"

python save_predictions.py $data_source $data_type 1 -f "$feature_suffix" -m "$model_suffix" -s "$save_format" -t "$top_k"
python fea_gen_kp_redis.py 0 2000000 $data_source $data_type debug basic_fea_coh${feature_suffix}_global coh${model_suffix}

for i in $(seq 2 4) ; do
  python save_predictions.py $data_source $data_type $i -f "$feature_suffix" -m "$model_suffix" -s "$save_format" -t "$top_k"
  python fea_gen_kp_redis.py 0 2000000 $data_source $data_type debug basic_fea_coh${feature_suffix}_global$i coh${model_suffix}
done
