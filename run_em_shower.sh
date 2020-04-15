#!/bin/bash
set -x
lr=${lr:-1e-4}
num_layers_emulsion=${num_layers_emulsion:-3}
num_layers_edge_conv=${num_layers_edge_conv:-5}
hidden_dim=${hidden_dim:-32}
output_dim=${output_dim:-32}
outer_optimization=${outer_optimization:-True}
use_scheduler=${use_scheduler:-False}
while [ $# -gt 0 ]; do
   if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
   fi
  shift
done
python training_classifier.py --datafile ./data/train_200_preprocessed.pt --epochs 4000 --learning_rate $lr \
--num_layers_emulsion $num_layers_emulsion --num_layers_edge_conv $num_layers_edge_conv \
--hidden_dim $hidden_dim --output_dim $output_dim --graph_embedder GraphNN_KNN_v1 --edge_classifier EdgeClassifier_v1 \
--project_name em_showers_network_training --workspace SchattenGenie --outer_optimization $outer_optimization \
--use_scheduler $use_scheduler