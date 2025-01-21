# Code for Hidden Node Discovery in Unknown Social Networks (ICWSM'25)
## About
This is a repository for hidden node discovery algorithms used in our following paper that will be presented at ICWSM'25.
> Sho Tsugawa, and Hiroyuki Ohsaki, "Exploring Unknown Social Networks for Discovering Hidden Nodes" Proceedings of the 19th International AAAI Conference on Web and Social Media (ICWSM 2025)

## Requirement
* This project uses code from [Python3 implemetation of DeepGL](https://github.com/takanori-fujiwara/deepgl), licensed under GPL-3.0.
* Python3
* graph-tool (https://graph-tool.skewed.de/)
* networkx

## Datasets
* 'data' directory contains Enron, Epinion, and Facebook networks used in our paper.
* Note that Twitter network is not included due to the Twitter's licence policy.

## Usage
### Preparing Peripheral Node Discovery
 ```bash
python3 make_periphery_data.py data/facebook
```
Label data will be generated under data directory.

### Preparing Sybil Node Discovery
 ```bash
python3 make_sybil_data.py data/facebook
```
Label and network data will be generated under data directory.

### Running Hidden Node Discovery
 ```bash
python3 hidden_node_discovery_bandit.py $net $label $train $init $batch $budget $setting $algorithms
```
#### Argument
* $net: network file name (e.g., data/facebook)
* $label: label file name (e.g., data/facebook.label)
* $train: Number of nodes used for model training in initial graph
* $init: Number of nodes in initial graph that are not used for model training
* $batch: Number of queries in each round
* $budget: Number of nodes to be queried
* $setting: File for DeepGL parameters (see 'deepgl_setting' in our repository)
* $algorithms: ML models or heuristcs used for node discovery
  * lgbm:LightGBM with basic features
  * lgbm_deep:LightGBM with DeepGL features
  * rf:RandomForest with basic features
  * rf_deep:FandomForest with DeepGL features
  * lr: Logistic Regression with basic features
  * lr_deep:Logistic Regression with DeepGL features
  * degree: MOD heuristic
  * nei: TN heuristic
When you specify multiple algorithms, the bandit algorithm is used for combining multiple strategies.
Please see our paper for more detailed explanation about the parameters.

### Reproducing the main results in our paper

 ```bash
sh run_sybil.sh
sh run_periphery.sh
```

## Output
The file with prefix "res" contains the results.
Each line contains:
> [Number of sampled nodes] [Number of Discovered Targets] [Number of Observed Nodes]

  

  
