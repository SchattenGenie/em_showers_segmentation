#!/bin/bash
set -x
num_layers_emulsion=${num_layers_emulsion:-3}
num_layers_edge_conv=${num_layers_edge_conv:-5}
hidden_dim=${hidden_dim:-32}
output_dim=${output_dim:-32}
key=${key:-1}
cl_size=${cl_size:-40}
threshold=${threshold:-0.5}
while [ $# -gt 0 ]; do
   if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
   fi
  shift
done
python clustering.py --datafile ./data/train_200_preprocessed.pt \
--num_layers_emulsion $num_layers_emulsion --num_layers_edge_conv $num_layers_edge_conv \
--hidden_dim $hidden_dim --output_dim $output_dim --graph_embedder GraphNN_KNN_v1 --edge_classifier EdgeClassifier_v1 \
--project_name em_showers_network_clustering --workspace SchattenGenie \
--graph_embedder_weights graph_embedder_train_200_preprocessed_"$key".pt \
--edge_classifier_weights edge_classifier_train_200_preprocessed_"$key".pt \
--cl_size $cl_size --min_cl $cl_size --threshold $threshold
