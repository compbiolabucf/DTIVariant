import re,sys,os
from utils import utils
from utils.preprocess import PREPROCESS 
import numpy as np
import pandas as pd 
import torch
import warnings
warnings.filterwarnings("ignore", category=Warning, module="torchvision")
warnings.filterwarnings('ignore')
import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig,OmegaConf 
from typing import Any, List, Optional, Tuple
from pytorch_lightning.loggers import TensorBoardLogger
import ray
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray import tune
import yaml
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
import multiprocessing
from multiprocessing import Manager
from datamodule.dataloader_GAT2 import UNIDataModule
os.environ["TOKENIZERS_PARALLELISM"] = "false"

"""
define device for featurizer that's outside the lightning module
"""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = 'cpu'
def train(cfg,dataset,shared_metrics,tune=False):
    tb_logger = TensorBoardLogger('../logger_DTI/tb_logs', name=cfg['logger']['name'])
    
    # Init datamodule
    # data_module: LightningModule = hydra.utils.instantiate(
    #     cfg['datamodule'],cfg, dataset, _recursive_=False
    # )
    data_module = UNIDataModule(cfg['datamodule'],dataset,cfg['datamodule']['dm_cfg'])
    # Init lightning model
    model: LightningModule = hydra.utils.instantiate(
        cfg['module'],cfg,dataset, _recursive_=False
    )
    # Init callbacks (early stopping, checkpointing)
    callbacks: List[Callback] = utils.instantiate_callbacks(
        cfg['callbacks']
    )

    if tune:
        metrics = {"auc": "val_auc","loss": "val_loss"}
        callbacks.append(TuneReportCallback(metrics, on="validation_end"))
        trainer = pl.Trainer(accelerator='gpu', devices=[1], max_epochs=cfg['trainer']['max_epochs'], logger=tb_logger,callbacks=callbacks, enable_progress_bar=False)
        trainer.fit(model, data_module)
    else:
        trainer = pl.Trainer(accelerator='gpu', devices=[0], max_epochs=cfg['trainer']['max_epochs'], logger=tb_logger,callbacks=callbacks,log_every_n_steps=5)
        trainer.fit(model, data_module)
        trainer.validate(model, data_module)
        trainer.test(model, data_module)
        shared_metrics["test_auc"]+= [model.test_auc.item()]
        shared_metrics["test_auprc"]+= [model.test_auprc.item()]
        shared_metrics["test_f1"]+= [model.test_f1.item()]

_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": "configs",
    "config_name": "config-name",
}


@hydra.main(**_HYDRA_PARAMS)
def main(cfg) -> Optional[float]:  
    """
    Any necessary precrocessing of the data to return
    X_drug: nx1 pd.DataFrame, index=drug names, Column 1=SMILES sequence. 
    X_target: mx1 pd.DataFrame, index=target names, Column 1=protein sequence. 
    DTI: mxn (available) pd.DataFrame, index: 0-mxn, Column 1=Drug names matching X_drug index, Column 2=Target names matching
    X_target index, Column 3= interaction label (0,1)
    """
    """
    If any DDI is to be used other than LM encoding based DDI, it can be returned here. 
    Deafult in get_ddi is raw sequence based DDI.
    For encoding based DDI, skip it. 
    """
    #ddi,skipped = utils.get_ddi(X_drug)
    
    logger, logger_dir = utils.get_logger(OmegaConf.to_container(cfg))
    new_dir = logger_dir.split('run')[0]
    
    """
    Load or generate/save drug and target LM encodings, For any other encoding, load X_drug and X_target accordingly
    """


    manager = Manager()
    shared_metrics = manager.dict()
    shared_metrics["test_auc"],shared_metrics["test_auprc"],shared_metrics["test_f1"] = [], [], []

    if cfg['tuning']['param_search']['tune']:
        optuna_search = OptunaSearch(
            metric="auc",
            mode="max",
        )

        asha_scheduler = ASHAScheduler(
            time_attr='training_iteration',
            metric='auc',
            mode='max',
            max_t=100,
            grace_period=10,
        )


        train_df = pd.read_csv(f"datasets/{cfg['preprocess']['name']}/custom_train_{cfg['variant_pred']['train']}.csv")
        valid_df = pd.read_csv(f"datasets/{cfg['preprocess']['name']}/custom_valid_default.csv")
        print("Testing On ", f"datasets/{cfg['preprocess']['name']}/custom_test_{cfg['variant_pred']['test']}.csv")
        test_df = pd.read_csv(f"datasets/{cfg['preprocess']['name']}/custom_test_{cfg['variant_pred']['test']}.csv")
        dataset = {'train':train_df, 'val':valid_df,'test':test_df}
        cfg = utils.setup_config_tune(cfg['tuning']['param_search']['search_space'],cfg)
        ray.init()
        trainable = tune.with_parameters(train, dataset=dataset, shared_metrics=None, tune=True)
        analysis = tune.run(
            trainable,
            local_dir="saved_models/",
            resources_per_trial={"gpu": 0.5},
            config=cfg,
            num_samples=100,
            search_alg=optuna_search,
            scheduler=asha_scheduler,
        )
        best_trial = analysis.get_best_trial("auc", mode="max")
        print("Best Hyperparameters:")
        print(best_trial.config)
        train(best_trial.config,dataset,shared_metrics)
        ray.shutdown()
    else:
        """
        set best parameters file for the experiment and update model configs from that file 
        """
        cfg = utils.update_best_param(cfg)
        
        pl.seed_everything(seed=42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        train_df = pd.read_csv(f"datasets/{cfg['preprocess']['name']}/custom_train_{cfg['variant_pred']['train']}.csv")
        valid_df = pd.read_csv(f"datasets/{cfg['preprocess']['name']}/custom_valid_default.csv")
        print("Testing On ", f"datasets/{cfg['preprocess']['name']}/custom_test_{cfg['variant_pred']['test']}.csv")
        test_df = pd.read_csv(f"datasets/{cfg['preprocess']['name']}/custom_test_{cfg['variant_pred']['test']}.csv")
        print("TEST SIZE ", test_df.shape)
        dataset = {'train':train_df, 'val':valid_df,'test':test_df}

        train(cfg,dataset,shared_metrics)
        
        test_auc = [shared_metrics["test_auc"]]
        test_auprc = [shared_metrics["test_auprc"]]
        test_f1 = [shared_metrics["test_f1"]]
        logger.info(f'Test AUC: {np.mean(test_auc):.4f}')
        logger.info(f'Test AUPRC: {np.mean(test_auprc):.4f}')
        logger.info(f'Test F1: {np.mean(test_f1):.4f}')
        new_dir = f'{new_dir}/run_{np.mean(test_auc):.4f}/'
        os.rename(logger_dir, new_dir)
            
        
if __name__ == "__main__":
    main()

  
