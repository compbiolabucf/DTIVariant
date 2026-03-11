import torch, os
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import DataLoader
from torch import nn 
import copy

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve, confusion_matrix, precision_score, recall_score, auc
from sklearn.model_selection import KFold
torch.manual_seed(1)    # reproducible torch:2 np:3
np.random.seed(1)

from config import BIN_config_DBPE
from models import BIN_Interaction_Flat
from stream import BIN_Data_Encoder

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:1" if use_cuda else "cpu")

from utils import test
import variant_utils

def main(fold_n, lr, dataFolder, file, train_type, test_type, modify=False):
    config = BIN_config_DBPE()
    
    lr = lr
    BATCH_SIZE = config['batch_size']
    train_epoch = 40
    
    loss_history = []
    
    model = BIN_Interaction_Flat(**config)
    
    model = model.to(device)

    # if torch.cuda.device_count() > 1:
    #     # print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model, dim = 0)
            
    opt = torch.optim.Adam(model.parameters(), lr = lr)
    #opt = torch.optim.SGD(model.parameters(), lr = lr, momentum=0.9)
    
    # print('--- Data Preparation ---')
    
    params = {'batch_size': BATCH_SIZE,'shuffle': True,'drop_last': True}

    # dataFolder = './dataset/BindingDB'
    # dataFolder = './dataset/DrugBank/balanced/warm/split0'
    df_train = pd.read_csv(dataFolder + '/train.csv').dropna().reset_index(drop=True)
    df_val = pd.read_csv(dataFolder + '/val.csv').dropna().reset_index(drop=True)
    df_test = pd.read_csv(dataFolder + '/test.csv').dropna().reset_index(drop=True)
    
    print("BEFORE SHAPES: ", df_train.shape, df_val.shape, df_test.shape)

    if splitting_strat == 'cold_target' and pred:
        print("coldtarget")
        df_train, df_test = variant_utils.cold_protein(df_train, df_test, dir, train_type)
    elif pred:
        print("warm pred")
        df_train = variant_utils.replace_train(df_train, dir, train_type)
        df_train = df_train.reset_index(drop=True)
    if pred:
        print("test pred")
        df_test = variant_utils.replace_test(df_test, dir, test_type, modify)
        df_test = df_test.reset_index(drop=True)

    
    print("AFTER SHAPES: ", df_train.shape, df_val.shape, df_test.shape)
    dirr = "/data/istiaq/work/projects/dti/DTIVariant/Moltrans/dataset/DrugBank/"
    train_def = dirr + "custom_train_default.csv"
    train_fur = dirr + "custom_train_furthest.csv"
    test_def = dirr + "custom_test_default.csv"
    test_fur = dirr + "custom_test_furthest.csv"
    val_def = dirr + "custom_valid_default.csv"

    print(train_def, test_def, val_def, dir)
    df_train.to_csv(train_fur, index = None)
    df_test.to_csv(test_fur, index = None)
    # df_val.to_csv(val_def, index = None)
    exit(0)

    training_set = BIN_Data_Encoder(df_train.index.values, df_train.Label.values, df_train)
    params = {'batch_size': min(BATCH_SIZE,training_set.__len__()),'shuffle': True,'drop_last': True}
    training_generator = data.DataLoader(training_set, **params)
    
    validation_set = BIN_Data_Encoder(df_val.index.values, df_val.Label.values, df_val)
    params = {'batch_size': min(BATCH_SIZE,validation_set.__len__()),'shuffle': True,'drop_last': True}
    validation_generator = data.DataLoader(validation_set, **params)
    
    testing_set = BIN_Data_Encoder(df_test.index.values, df_test.Label.values, df_test)
    params = {'batch_size': min(BATCH_SIZE,testing_set.__len__()),'shuffle': True,'drop_last': True}
    testing_generator = data.DataLoader(testing_set, **params)
    
    # early stopping
    max_auc = 0
    model_max = copy.deepcopy(model)
    
    # print('--- Go for Training ---')
    torch.backends.cudnn.benchmark = True
    for epo in tqdm(range(train_epoch)):
        model.train()
        for i, (d, p, d_mask, p_mask, label) in enumerate(training_generator):
            score = model(d.long().to(device), p.long().to(device), d_mask.long().to(device), p_mask.long().to(device))

            label = Variable(torch.from_numpy(np.array(label)).float()).to(device)
            
            loss_fct = torch.nn.BCELoss()
            m = torch.nn.Sigmoid()
            n = torch.squeeze(m(score))
            
            loss = loss_fct(n, label)
            loss_history.append(loss)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            # if (i % 100 == 0):
            #     print('Training at Epoch ' + str(epo + 1) + ' iteration ' + str(i) + ' with loss ' + str(loss.cpu().detach().numpy()))
            
        # every epoch test
        with torch.set_grad_enabled(False):
            auc, auprc, f1, tn, fp, fn, tp = test(validation_generator, model,device=device)
            if auc > max_auc:
                model_max = copy.deepcopy(model)
                max_auc = auc
            
            # print('Validation at Epoch '+ str(epo + 1) + ' , AUROC: '+ str(auc) + ' , AUPRC: ' + str(auprc) + ' , F1: '+str(f1))
        v_auc = auc
        v_aucprc = auprc
        # print('--- Go for Testing ---')
        with torch.set_grad_enabled(False):
            auc, auprc, f1, tn, fp, fn, tp = test(testing_generator, model_max,device=device)
            # print('Testing AUROC: ' + str(auc) + ' , AUPRC: ' + str(auprc) + ' , F1: '+str(f1) + ' , Test loss: '+str(loss))
        t_auc = auc
        t_aucprc = auprc
        file.write(f"|{v_auc},{v_aucprc},{t_auc},{t_aucprc},{f1},{tn},{fp},{fn},{tp}")
    return model_max, loss_history

dataset = "DrugBank"
# dataset = "DrugBank"
splitting_strat = "warm"
logfiles = f"./logs/variant2/{dataset}/balanced/{splitting_strat}/split"
datadirs = f"./dataset/{dataset}/balanced/{splitting_strat}/split"
dir = "dataset/variants/drugbank_rep_2_vars.pkl"
pred = True
train_type = "furthest"
test_type = "furthest"
modify = True




# data_dir = "./dataset/BindingDB/balanced/warm/"
# save_path = f"./logs/BindingDB/balanced/warm/"

print("DATASETTTTT ",datadirs)

# for train_type,test_type in zip(["default","default","default","default"],["default", "random","closest", "furthest"]):
split  = 0
print("SPILT ", split, train_type, test_type)
save_path = logfiles + str(split) +"/"
data_file = datadirs + str(split) +"/"
os.makedirs(save_path,exist_ok=True)
save_path = os.path.join(save_path, train_type+test_type+"_mod" if modify else "" +"_out.csv")
file = open(save_path, "a")
s = time()
dataFolder = data_file
file.write(f"{dataFolder}{train_type}{test_type}")
model_max, loss_history = main(1, 5e-6, dataFolder, file,train_type,test_type,modify)
file.write(f"\n")
file.close()
e = time()
print(e-s)
print(file)