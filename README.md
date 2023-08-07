# SAD: Semi-Supervised Anomaly Detection on Dynamic Graphs

[PyTorch implementation of the paper "SAD: Semi-Supervised Anomaly Detection on Dynamic Graphs".](https://arxiv.org/abs/2305.13573)

#  Requirments
+ torch==1.10.1+cu111
+ torchvision==0.9.1+cu111
+ torch-geometric==2.0.4
+ torch-scatter==2.0.9
+ torch-sparse==0.6.12
+ scikit-learn==0.23.2

# Preprocessing
## Dataset
Download data.csv into file './dataset/'  
  
[Wikipedia](http://snap.stanford.edu/jodie/wikipedia.csv)  
[Reddit](http://snap.stanford.edu/jodie/reddit.csv)
[Mooc](http://snap.stanford.edu/jodie/mooc.csv)
## Preprocessing
We use the data processing method of the reference [TGAT](https://openreview.net/pdf?id=rJeW1yHYwH), [repo](https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs#inductive-representation-learning-on-temporal-graphs-iclr-2020).  
We use then dense npy format to save the features in binary format. If edge features or nodes features are absent, it will replaced by a vector of zeros.  (While preprocess mooc data, ```rand_feat = np.zeros((max_idx + 1, 172))```, ```172``` need change to ```4```)
```python
python build_dataset_graph.py --data wikipedia --bipartite
python build_dataset_graph.py --data reddit --bipartite
python build_dataset_graph.py --data mooc --bipartite
```
# Model Training

Training the SAD Graph network based on all black samples.
```python
python train.py --data_set wikipedia --anomaly_alpha 1e-1 --supc_alpha 5e-3
```
Training the SAD Graph network based on half of black samples.
```python
python train.py --data_set wikipedia --anomaly_alpha 1e-1 --supc_alpha 5e-3 --mask_label --mask_ratio 0.5
```

# Model Structure
![image](https://github.com/D10Andy/SAD/blob/main/model_structure.jpg)
