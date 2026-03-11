python run.py --config-name drugbank_train_GAT.yaml "tuning.param_search.tune=False" "datamodule.splitting.balanced=True" "datamodule.splitting.splitting_strategy=random"



python run.py --config-name bindingDB_train_GAT.yaml "tuning.param_search.tune=False" "datamodule.splitting.balanced=True" "datamodule.splitting.splitting_strategy=random"


train - default 
test - default, random, closest, furthest
