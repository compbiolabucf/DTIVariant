import torch,sys,hydra
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from utils import utils, variant_utils
import numpy as np 
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt

# pytorch datalaoder
class MyDataset(Dataset):
    def __init__(self, drug, target, DTI, test=False):
        self.test = test
        self.drug = drug
        self.target = target
        self.DTI = DTI
        # print("COlumns ")
        # print(self.DTI.columns)
        # print(self.DTI.head())
        # print(self.target.head())
        # print(self.drug.head())
        # exit(0)
        self.device = "cpu"
        self.prot_featurizer = hydra.utils.instantiate(  {"_target_": "module.featurizer.prot_featurizer.esm_featurizer.ESMFEATURE"}, self.device, _recursive_=False)
        self.drug_featurizer = hydra.utils.instantiate(  {"_target_": "module.featurizer.drug_featurizer.chembert_featurizer.CHEMFEATURE"}, self.device, _recursive_=False)
        self.residue_cut_count = 0
    def __getitem__(self, index):
        y = self.DTI.iloc[index, 2] # label
        drug_index = self.DTI.iloc[index]['Drug_ID']
        target_index = self.DTI.iloc[index]['Prot_ID']
        smile = self.drug[self.drug['Drug_ID']==drug_index]["SMILES"].values[0]
        seq = self.target[self.target['Prot_ID']==target_index]["SEQ"].values[0]
        if self.test:
            x1 = self.drug[self.drug['Drug_ID']==drug_index]["Drug_Feat"].values[0]
            # x2 = self.target[self.target['Prot_ID']==target_index]["Prot_Feat"].values[0]
            yes,seq2 = variant_utils.get_modified_seq(seq,smile)
            x2 = self.prot_featurizer.get_representations([seq2])[0]
            self.residue_cut_count += yes
            print(self.residue_cut_count, "out of ", self.DTI.shape)
            # exit(0)
        else:
            x1 = self.drug[self.drug['Drug_ID']==drug_index]["Drug_Feat"].values[0]
            x2 = self.target[self.target['Prot_ID']==target_index]["Prot_Feat"].values[0]
            # x3 = self.prot_featurizer.get_representations([seq])[0]
            # print("2"*50, x3.shape)
            # exit(0)
        try:
            pdb_id = self.DTI.iloc[index, 3]
        except:
            pdb_id = ""
        if(len(pdb_id)>0):
            pdb_id = pdb_id[0]
        else:
            pdb_id = ""
        return torch.tensor(x1).float(), torch.tensor(x2).float(), torch.tensor(y).float(), drug_index, target_index, smile, seq, pdb_id
        # drug_feat,prot_feat,label,drugID,protID,smile,seq,pdb
    def __len__(self):
        return len(self.DTI)


class UNIDataModule(pl.LightningDataModule):
    def __init__(self,config,dataset,dm_cfg,splitting,serializer):
        super().__init__()
        self.X_drug = dataset['X_drug']
        self.X_target = dataset['X_target']
        self.X_target_orig = self.X_target.copy()
        self.train_ind = dataset['train']
        self.val_ind = dataset['val']
        self.test_ind = dataset['test']    
        self.batch_size = dm_cfg['batch_size']
        self.num_workers = dm_cfg['num_workers']
        self.config = config
    
    def prepare_data(self):
        if self.config['datamodule']['splitting']['splitting_strategy'] == 'cold_target' and self.config['variant_pred']['pred']:
            self.X_target, self.train_ind, self.test_ind = variant_utils.cold_protein(self.X_target_orig, self.train_ind,self.test_ind,self.config['variant_pred']['dir'])
        elif self.config['variant_pred']['pred']:
            self.X_target, self.train_ind = variant_utils.replace_train(self.X_target_orig, self.train_ind, self.config['variant_pred']['dir'], self.config['variant_pred']['train'] )
        if self.config['variant_pred']['pred']:
            self.X_target_rep,self.test_ind = variant_utils.replace_test(self.X_target_orig,self.test_ind,self.config['variant_pred']['dir'], self.config['variant_pred']['test'])
        
    def setup(self, stage):
        print("Number of samples in training",len(self.train_ind))
        print(f'Number of samples in test: {len(self.test_ind)}')
        self.train_dataset = MyDataset(self.X_drug, self.X_target, self.train_ind)
        self.val_dataset = MyDataset(self.X_drug, self.X_target, self.val_ind)
        self.test_dataset = MyDataset(self.X_drug, self.X_target, self.test_ind, test=True)
        if self.config['variant_pred']['pred']:
            self.test_dataset_rep = MyDataset(self.X_drug, self.X_target_rep, self.test_ind, test=True)

    def train_dataloader(self):
        # REQUIRED
        return DataLoader(self.train_dataset, batch_size=self.batch_size , shuffle=True, num_workers=self.num_workers,
                          pin_memory=True, drop_last=True)

    def val_dataloader(self):
        # OPTIONAL
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          pin_memory=True, drop_last=False)

    def test_dataloader(self):
        # OPTIONAL
        
        loader1 = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                         pin_memory=True, drop_last=False)
        if self.config['variant_pred']['pred']:
            loader2= DataLoader(self.test_dataset_rep, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                                pin_memory=True, drop_last=False)
            return loader2
        return loader1  #[laoder1,laoder2]
