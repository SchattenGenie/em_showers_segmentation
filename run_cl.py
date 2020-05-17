import subprocess
import numpy as np
import shlex
import time
import os
from tqdm import tqdm

def main():
    processes = []
    commands = """
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 67971cd97d124f368379ccb3917811c6 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold -1.40 --min_samples_core 0
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 380d538fcb9c4e7ba158bdb659e65543 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold -1.40 --min_samples_core 0
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 579d9fc429274d72a9be79c244eed210 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold -1.40 --min_samples_core 0
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 67971cd97d124f368379ccb3917811c6 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold -1.40 --min_samples_core 2
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 380d538fcb9c4e7ba158bdb659e65543 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold -1.40 --min_samples_core 2
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 579d9fc429274d72a9be79c244eed210 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold -1.40 --min_samples_core 2
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 67971cd97d124f368379ccb3917811c6 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold -1.40 --min_samples_core 4
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 380d538fcb9c4e7ba158bdb659e65543 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold -1.40 --min_samples_core 4
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 579d9fc429274d72a9be79c244eed210 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold -1.40 --min_samples_core 4
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 67971cd97d124f368379ccb3917811c6 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold -1.40 --min_samples_core 6
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 380d538fcb9c4e7ba158bdb659e65543 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold -1.40 --min_samples_core 6
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 579d9fc429274d72a9be79c244eed210 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold -1.40 --min_samples_core 6
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 67971cd97d124f368379ccb3917811c6 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold -1.40 --min_samples_core 8
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 380d538fcb9c4e7ba158bdb659e65543 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold -1.40 --min_samples_core 8
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 579d9fc429274d72a9be79c244eed210 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold -1.40 --min_samples_core 8
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 67971cd97d124f368379ccb3917811c6 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold -1.00 --min_samples_core 0
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 380d538fcb9c4e7ba158bdb659e65543 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold -1.00 --min_samples_core 0
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 579d9fc429274d72a9be79c244eed210 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold -1.00 --min_samples_core 0
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 67971cd97d124f368379ccb3917811c6 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold -1.00 --min_samples_core 2
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 380d538fcb9c4e7ba158bdb659e65543 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold -1.00 --min_samples_core 2
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 579d9fc429274d72a9be79c244eed210 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold -1.00 --min_samples_core 2
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 67971cd97d124f368379ccb3917811c6 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold -1.00 --min_samples_core 4
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 380d538fcb9c4e7ba158bdb659e65543 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold -1.00 --min_samples_core 4
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 579d9fc429274d72a9be79c244eed210 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold -1.00 --min_samples_core 4
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 67971cd97d124f368379ccb3917811c6 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold -1.00 --min_samples_core 6
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 380d538fcb9c4e7ba158bdb659e65543 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold -1.00 --min_samples_core 6
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 579d9fc429274d72a9be79c244eed210 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold -1.00 --min_samples_core 6
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 67971cd97d124f368379ccb3917811c6 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold -1.00 --min_samples_core 8
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 380d538fcb9c4e7ba158bdb659e65543 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold -1.00 --min_samples_core 8
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 579d9fc429274d72a9be79c244eed210 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold -1.00 --min_samples_core 8
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 67971cd97d124f368379ccb3917811c6 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold -0.60 --min_samples_core 0
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 380d538fcb9c4e7ba158bdb659e65543 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold -0.60 --min_samples_core 0
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 579d9fc429274d72a9be79c244eed210 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold -0.60 --min_samples_core 0
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 67971cd97d124f368379ccb3917811c6 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold -0.60 --min_samples_core 2
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 380d538fcb9c4e7ba158bdb659e65543 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold -0.60 --min_samples_core 2
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 579d9fc429274d72a9be79c244eed210 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold -0.60 --min_samples_core 2
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 67971cd97d124f368379ccb3917811c6 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold -0.60 --min_samples_core 4
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 380d538fcb9c4e7ba158bdb659e65543 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold -0.60 --min_samples_core 4
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 579d9fc429274d72a9be79c244eed210 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold -0.60 --min_samples_core 4
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 67971cd97d124f368379ccb3917811c6 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold -0.60 --min_samples_core 6
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 380d538fcb9c4e7ba158bdb659e65543 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold -0.60 --min_samples_core 6
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 579d9fc429274d72a9be79c244eed210 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold -0.60 --min_samples_core 6
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 67971cd97d124f368379ccb3917811c6 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold -0.60 --min_samples_core 8
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 380d538fcb9c4e7ba158bdb659e65543 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold -0.60 --min_samples_core 8
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 579d9fc429274d72a9be79c244eed210 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold -0.60 --min_samples_core 8
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 67971cd97d124f368379ccb3917811c6 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold -0.20 --min_samples_core 0
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 380d538fcb9c4e7ba158bdb659e65543 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold -0.20 --min_samples_core 0
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 579d9fc429274d72a9be79c244eed210 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold -0.20 --min_samples_core 0
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 67971cd97d124f368379ccb3917811c6 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold -0.20 --min_samples_core 2
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 380d538fcb9c4e7ba158bdb659e65543 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold -0.20 --min_samples_core 2
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 579d9fc429274d72a9be79c244eed210 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold -0.20 --min_samples_core 2
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 67971cd97d124f368379ccb3917811c6 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold -0.20 --min_samples_core 4
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 380d538fcb9c4e7ba158bdb659e65543 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold -0.20 --min_samples_core 4
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 579d9fc429274d72a9be79c244eed210 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold -0.20 --min_samples_core 4
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 67971cd97d124f368379ccb3917811c6 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold -0.20 --min_samples_core 6
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 380d538fcb9c4e7ba158bdb659e65543 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold -0.20 --min_samples_core 6
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 579d9fc429274d72a9be79c244eed210 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold -0.20 --min_samples_core 6
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 67971cd97d124f368379ccb3917811c6 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold -0.20 --min_samples_core 8
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 380d538fcb9c4e7ba158bdb659e65543 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold -0.20 --min_samples_core 8
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 579d9fc429274d72a9be79c244eed210 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold -0.20 --min_samples_core 8
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 67971cd97d124f368379ccb3917811c6 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 0.20 --min_samples_core 0
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 380d538fcb9c4e7ba158bdb659e65543 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 0.20 --min_samples_core 0
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 579d9fc429274d72a9be79c244eed210 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 0.20 --min_samples_core 0
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 67971cd97d124f368379ccb3917811c6 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 0.20 --min_samples_core 2
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 380d538fcb9c4e7ba158bdb659e65543 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 0.20 --min_samples_core 2
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 579d9fc429274d72a9be79c244eed210 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 0.20 --min_samples_core 2
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 67971cd97d124f368379ccb3917811c6 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 0.20 --min_samples_core 4
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 380d538fcb9c4e7ba158bdb659e65543 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 0.20 --min_samples_core 4
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 579d9fc429274d72a9be79c244eed210 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 0.20 --min_samples_core 4
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 67971cd97d124f368379ccb3917811c6 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 0.20 --min_samples_core 6
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 380d538fcb9c4e7ba158bdb659e65543 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 0.20 --min_samples_core 6
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 579d9fc429274d72a9be79c244eed210 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 0.20 --min_samples_core 6
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 67971cd97d124f368379ccb3917811c6 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 0.20 --min_samples_core 8
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 380d538fcb9c4e7ba158bdb659e65543 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 0.20 --min_samples_core 8
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 579d9fc429274d72a9be79c244eed210 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 0.20 --min_samples_core 8
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 67971cd97d124f368379ccb3917811c6 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 0.60 --min_samples_core 0
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 380d538fcb9c4e7ba158bdb659e65543 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 0.60 --min_samples_core 0
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 579d9fc429274d72a9be79c244eed210 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 0.60 --min_samples_core 0
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 67971cd97d124f368379ccb3917811c6 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 0.60 --min_samples_core 2
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 380d538fcb9c4e7ba158bdb659e65543 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 0.60 --min_samples_core 2
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 579d9fc429274d72a9be79c244eed210 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 0.60 --min_samples_core 2
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 67971cd97d124f368379ccb3917811c6 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 0.60 --min_samples_core 4
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 380d538fcb9c4e7ba158bdb659e65543 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 0.60 --min_samples_core 4
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 579d9fc429274d72a9be79c244eed210 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 0.60 --min_samples_core 4
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 67971cd97d124f368379ccb3917811c6 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 0.60 --min_samples_core 6
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 380d538fcb9c4e7ba158bdb659e65543 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 0.60 --min_samples_core 6
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 579d9fc429274d72a9be79c244eed210 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 0.60 --min_samples_core 6
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 67971cd97d124f368379ccb3917811c6 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 0.60 --min_samples_core 8
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 380d538fcb9c4e7ba158bdb659e65543 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 0.60 --min_samples_core 8
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 579d9fc429274d72a9be79c244eed210 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 0.60 --min_samples_core 8
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 67971cd97d124f368379ccb3917811c6 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 1.00 --min_samples_core 0
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 380d538fcb9c4e7ba158bdb659e65543 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 1.00 --min_samples_core 0
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 579d9fc429274d72a9be79c244eed210 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 1.00 --min_samples_core 0
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 67971cd97d124f368379ccb3917811c6 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 1.00 --min_samples_core 2
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 380d538fcb9c4e7ba158bdb659e65543 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 1.00 --min_samples_core 2
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 579d9fc429274d72a9be79c244eed210 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 1.00 --min_samples_core 2
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 67971cd97d124f368379ccb3917811c6 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 1.00 --min_samples_core 4
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 380d538fcb9c4e7ba158bdb659e65543 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 1.00 --min_samples_core 4
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 579d9fc429274d72a9be79c244eed210 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 1.00 --min_samples_core 4
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 67971cd97d124f368379ccb3917811c6 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 1.00 --min_samples_core 6
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 380d538fcb9c4e7ba158bdb659e65543 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 1.00 --min_samples_core 6
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 579d9fc429274d72a9be79c244eed210 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 1.00 --min_samples_core 6
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 67971cd97d124f368379ccb3917811c6 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 1.00 --min_samples_core 8
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 380d538fcb9c4e7ba158bdb659e65543 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 1.00 --min_samples_core 8
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 579d9fc429274d72a9be79c244eed210 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 1.00 --min_samples_core 8
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 67971cd97d124f368379ccb3917811c6 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 1.40 --min_samples_core 0
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 380d538fcb9c4e7ba158bdb659e65543 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 1.40 --min_samples_core 0
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 579d9fc429274d72a9be79c244eed210 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 1.40 --min_samples_core 0
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 67971cd97d124f368379ccb3917811c6 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 1.40 --min_samples_core 2
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 380d538fcb9c4e7ba158bdb659e65543 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 1.40 --min_samples_core 2
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 579d9fc429274d72a9be79c244eed210 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 1.40 --min_samples_core 2
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 67971cd97d124f368379ccb3917811c6 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 1.40 --min_samples_core 4
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 380d538fcb9c4e7ba158bdb659e65543 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 1.40 --min_samples_core 4
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 579d9fc429274d72a9be79c244eed210 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 1.40 --min_samples_core 4
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 67971cd97d124f368379ccb3917811c6 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 1.40 --min_samples_core 6
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 380d538fcb9c4e7ba158bdb659e65543 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 1.40 --min_samples_core 6
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 579d9fc429274d72a9be79c244eed210 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 1.40 --min_samples_core 6
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 67971cd97d124f368379ccb3917811c6 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 1.40 --min_samples_core 8
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 380d538fcb9c4e7ba158bdb659e65543 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 1.40 --min_samples_core 8
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 579d9fc429274d72a9be79c244eed210 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 1.40 --min_samples_core 8
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 67971cd97d124f368379ccb3917811c6 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 1.80 --min_samples_core 0
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 380d538fcb9c4e7ba158bdb659e65543 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 1.80 --min_samples_core 0
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 579d9fc429274d72a9be79c244eed210 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 1.80 --min_samples_core 0
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 67971cd97d124f368379ccb3917811c6 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 1.80 --min_samples_core 2
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 380d538fcb9c4e7ba158bdb659e65543 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 1.80 --min_samples_core 2
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 579d9fc429274d72a9be79c244eed210 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 1.80 --min_samples_core 2
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 67971cd97d124f368379ccb3917811c6 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 1.80 --min_samples_core 4
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 380d538fcb9c4e7ba158bdb659e65543 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 1.80 --min_samples_core 4
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 579d9fc429274d72a9be79c244eed210 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 1.80 --min_samples_core 4
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 67971cd97d124f368379ccb3917811c6 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 1.80 --min_samples_core 6
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 380d538fcb9c4e7ba158bdb659e65543 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 1.80 --min_samples_core 6
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 579d9fc429274d72a9be79c244eed210 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 1.80 --min_samples_core 6
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 67971cd97d124f368379ccb3917811c6 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 1.80 --min_samples_core 8
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 380d538fcb9c4e7ba158bdb659e65543 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 1.80 --min_samples_core 8
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 579d9fc429274d72a9be79c244eed210 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 1.80 --min_samples_core 8
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 67971cd97d124f368379ccb3917811c6 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 2.20 --min_samples_core 0
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 380d538fcb9c4e7ba158bdb659e65543 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 2.20 --min_samples_core 0
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 579d9fc429274d72a9be79c244eed210 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 2.20 --min_samples_core 0
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 67971cd97d124f368379ccb3917811c6 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 2.20 --min_samples_core 2
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 380d538fcb9c4e7ba158bdb659e65543 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 2.20 --min_samples_core 2
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 579d9fc429274d72a9be79c244eed210 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 2.20 --min_samples_core 2
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 67971cd97d124f368379ccb3917811c6 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 2.20 --min_samples_core 4
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 380d538fcb9c4e7ba158bdb659e65543 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 2.20 --min_samples_core 4
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 579d9fc429274d72a9be79c244eed210 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 2.20 --min_samples_core 4
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 67971cd97d124f368379ccb3917811c6 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 2.20 --min_samples_core 6
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 380d538fcb9c4e7ba158bdb659e65543 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 2.20 --min_samples_core 6
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 579d9fc429274d72a9be79c244eed210 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 2.20 --min_samples_core 6
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 67971cd97d124f368379ccb3917811c6 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 2.20 --min_samples_core 8
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 380d538fcb9c4e7ba158bdb659e65543 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 2.20 --min_samples_core 8
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 579d9fc429274d72a9be79c244eed210 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 2.20 --min_samples_core 8
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 67971cd97d124f368379ccb3917811c6 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 2.60 --min_samples_core 0
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 380d538fcb9c4e7ba158bdb659e65543 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 2.60 --min_samples_core 0
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 579d9fc429274d72a9be79c244eed210 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 2.60 --min_samples_core 0
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 67971cd97d124f368379ccb3917811c6 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 2.60 --min_samples_core 2
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 380d538fcb9c4e7ba158bdb659e65543 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 2.60 --min_samples_core 2
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 579d9fc429274d72a9be79c244eed210 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 2.60 --min_samples_core 2
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 67971cd97d124f368379ccb3917811c6 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 2.60 --min_samples_core 4
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 380d538fcb9c4e7ba158bdb659e65543 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 2.60 --min_samples_core 4
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 579d9fc429274d72a9be79c244eed210 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 2.60 --min_samples_core 4
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 67971cd97d124f368379ccb3917811c6 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 2.60 --min_samples_core 6
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 380d538fcb9c4e7ba158bdb659e65543 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 2.60 --min_samples_core 6
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 579d9fc429274d72a9be79c244eed210 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 2.60 --min_samples_core 6
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 67971cd97d124f368379ccb3917811c6 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 2.60 --min_samples_core 8
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 380d538fcb9c4e7ba158bdb659e65543 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 2.60 --min_samples_core 8
sbatch -c 1 -t 720 --gpus=0 run_clustering.sh --key 579d9fc429274d72a9be79c244eed210 --num_layers_edge_conv 5 --num_layers_emulsion 3 -cl_size 40 --min_cl 40 --threshold 2.60 --min_samples_core 8
"""
    commands = commands.split("\n")
    batch_size = 8
    for command in tqdm(commands):
        print(command)
        process = subprocess.Popen(command,
                                   shell=True,
                                   close_fds=True,
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL
                                   )
        processes.append(process)
        pr_count = subprocess.Popen("squeue | grep vbelavin | wc -l", shell=True, stdout=subprocess.PIPE)
        out, err = pr_count.communicate()
        if int(out) > batch_size:
            while int(out) > batch_size:
                print("Waiting... ")
                time.sleep(60)
                pr_count = subprocess.Popen("squeue | grep vbelavin | wc -l", shell=True, stdout=subprocess.PIPE)
                out, err = pr_count.communicate()

    for process in processes:
        print(process.pid)


if __name__ == "__main__":
    main()
