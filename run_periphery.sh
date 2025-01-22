setting="deepgl_setting"


###########
batch=100
budget=5000
init=200
train=200


net="data/facebook"
label="data/facebook-kcore.label"


python3 node_discovery_with_topology.py $net $label $train $init $batch $budget $setting lgbm

#TN
python3 hidden_node_discovery_bandit.py $net $label $train $init $batch $budget $setting nei
#MOD
python3 hidden_node_discovery_bandit.py $net $label $train $init $batch $budget $setting degree

python3 hidden_node_discovery_bandit.py $net $label $train $init $batch $budget $setting lgbm
python3 hidden_node_discovery_bandit.py $net $label $train $init $batch $budget $setting lgbm_deep
python3 hidden_node_discovery_bandit.py $net $label $train $init $batch $budget $setting lgbm lgbm_deep

python3 hidden_node_discovery_bandit.py $net $label $train $init $batch $budget $setting rf
python3 hidden_node_discovery_bandit.py $net $label $train $init $batch $budget $setting rf_deep

python3 hidden_node_discovery_bandit.py $net $label $train $init $batch $budget $setting lr
python3 hidden_node_discovery_bandit.py $net $label $train $init $batch $budget $setting lr_deep

###########

batch=1000
budget=80000
init=1000
train=1000

net="data/enron"
label="data/enron-kcore.label"

python3 node_discovery_with_topology.py $net $label $train $init $batch $budget $setting lgbm


#TN
python3 hidden_node_discovery_bandit.py $net $label $train $init $batch $budget $setting nei
#MOD
python3 hidden_node_discovery_bandit.py $net $label $train $init $batch $budget $setting degree
python3 hidden_node_discovery_bandit.py $net $label $train $init $batch $budget $setting lgbm
python3 hidden_node_discovery_bandit.py $net $label $train $init $batch $budget $setting lgbm_deep
python3 hidden_node_discovery_bandit.py $net $label $train $init $batch $budget $setting lgbm lgbm_deep

python3 hidden_node_discovery_bandit.py $net $label $train $init $batch $budget $setting rf
python3 hidden_node_discovery_bandit.py $net $label $train $init $batch $budget $setting rf_deep

python3 hidden_node_discovery_bandit.py $net $label $train $init $batch $budget $setting lr
python3 hidden_node_discovery_bandit.py $net $label $train $init $batch $budget $setting lr_deep
###########

batch=1000
budget=80000
init=1000
train=1000

net="data/epinion"
label="data/epinion-kcore.label"

#TN
python3 hidden_node_discovery_bandit.py $net $label $train $init $batch $budget $setting nei
#MOD
python3 hidden_node_discovery_bandit.py $net $label $train $init $batch $budget $setting degree
python3 node_discovery_with_topology.py $net $label $train $init $batch $budget $setting lgbm
python3 hidden_node_discovery_bandit.py $net $label $train $init $batch $budget $setting lgbm
python3 hidden_node_discovery_bandit.py $net $label $train $init $batch $budget $setting lgbm_deep
python3 hidden_node_discovery_bandit.py $net $label $train $init $batch $budget $setting lgbm lgbm_deep

python3 hidden_node_discovery_bandit.py $net $label $train $init $batch $budget $setting rf
python3 hidden_node_discovery_bandit.py $net $label $train $init $batch $budget $setting rf_deep

python3 hidden_node_discovery_bandit.py $net $label $train $init $batch $budget $setting lr
python3 hidden_node_discovery_bandit.py $net $label $train $init $batch $budget $setting lr_deep
