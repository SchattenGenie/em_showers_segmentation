# Electromagnetic Showers Segmentation
Segmentation of multiple overlapping electromagnetic showers(point clouds) with Graph Neural Networks.

## 1. Data preprocessing

Firstly one have to compile Cython extension for graph calculation:

```
cd tools/ && python setup_opera_distance_metric.py build_ext --inplace
```

And then one could run preprocessing of EM showers in pytorch-geometric graph format:

```
python graph_generation.py --root_file ./data/mcdata_taue2.root --output_file ./data/train.pt --knn True --k 10 --directed False --e 10.
```


## 2. Graph Neural Network training edge classifier

Next step is to train classier network, that is going to discriminated edges between those that connect nodes from the same class and those which belogs to different classes.

```
python training_classifier.py  --datafile ./data/train_3_directed.pt --epochs 1000 --learning_rate 1e-3 --output_dim 32 --graph_embedder GraphNN_KNN_v1 --edge_classifier EdgeClassifier_v1 --project_name em_showers_network_training --work_space schattengenie
```

## 3. Clustering of EM showers

Using networks weights from previous step we can perform clustering end estimate quality:

```
python clustering.py --datafile ./data/train.pt --project_name em_showers_clustering --work_space schattengenie --min_cl 40 --cl_size 40 --threshold 0.9
```

```bash
python clustering.py  --datafile ./data/train.pt --project_name em_showers_clustering --work_space schattengenie --baseline True --min_cl 40 --cl_size 40 --threshold 0.9
```
