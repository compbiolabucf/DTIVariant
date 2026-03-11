import numpy as np
import pandas as pd
import sys,ast 
import pickle

def replace_tx(X_target, test_ind,dir):
    print(test_ind.shape)
    variant_data = pd.read_csv(dir)
    ref = pd.read_csv('DB_X_target.csv')
    ref = ref.set_index(ref.columns[0])
    
    replaced_data = X_target.copy()
    
    for i in range(len(replaced_data)):
        prot=replaced_data.index[i]
        #find all rows from variant data with ProtID prot
        temp_variant = variant_data[variant_data['Prot_ID']==prot]
        temp_ref = ref[ref.index==prot]

        if len(temp_variant)>1:
            #choose one variant randomly
            temp_variant = temp_variant.sample(n=1)
            #replace the representation of prot with the variant
            replaced_data.iloc[i] = ast.literal_eval(temp_variant['variant_representations'].iloc[0])
        else:
            #remove DT pairs from test_ind as those can not be used to variant
            
            test_ind = test_ind[test_ind.iloc[:,1]!=i]

    print(test_ind.shape)
    # sys.exit()
    return replaced_data, test_ind

def replace_test(X_target_org, test_ind_orig, pickle_file, split):
    X_target = X_target_org.copy()
    test_ind = test_ind_orig.copy()
    print("Unique proteins in test_ind ", len(set(test_ind['Prot_ID'])))
    with open(pickle_file, "rb") as f:
        drugvar = pickle.load(f)
    replaced_prots = []
    remove_rows = []
    for i,row in test_ind.iterrows():
        idx = row['Prot_ID']
        protid = idx
        if(protid not in drugvar):
            remove_rows.append(i)
        elif(drugvar[protid]['len']>0):
            if split=='default':
                continue
            elif split=='closest':
                df = drugvar[protid]['var']
                max_row = df.loc[df['similarity'].idxmax()]
                if(max_row['similarity']>1500):
                    X_target.loc[protid]['Prot_Feat'] = max_row['feature'][0]
                    X_target.loc[protid]['SEQ'] = max_row['SEQ']
                    replaced_prots.append(protid)
                else:
                    remove_rows.append(i)
            elif split=='furthest':
                df = drugvar[protid]['var']
                max_row = df.loc[df['similarity'].idxmin()]
                if(max_row['similarity']<1000):
                    X_target.loc[protid]['Prot_Feat'] = max_row['feature'][0]
                    X_target.loc[protid]['SEQ'] = max_row['SEQ']
                    replaced_prots.append(protid)
                else:
                    remove_rows.append(i)
            elif split=='random':
                # Select random
                print("r", end="")
                df = drugvar[protid]['var']
                max_row = df.sample(n=1,random_state=1111)
                X_target.loc[protid]['Prot_Feat'] = max_row['feature'].values[0][0]
                X_target.loc[protid]['SEQ'] = max_row['SEQ']
                replaced_prots.append(protid)
        else:
            remove_rows.append(i)

    # indexes_to_remove = test_ind.iloc[remove_rows].index
    test_ind = test_ind.drop(remove_rows)

    print("Replaced proteins ", len(replaced_prots), "Unique ", len(set(replaced_prots)))
    print("Removed rows ", len(remove_rows))
    return X_target, test_ind

def replace_train(X_target_org, train_ind_orig, pickle_file, split):
    X_target = X_target_org.copy()
    train_ind = train_ind_orig.copy()
    print("Unique proteins in train_ind ", len(set(train_ind['Prot_ID'])), "total ", train_ind.shape)
    with open(pickle_file, "rb") as f:
        drugvar = pickle.load(f)
    replaced_prots = []
    remove_rows = []
    for i,row in train_ind.iterrows():
        idx = row['Prot_ID']
        protid = idx
        if(protid not in drugvar):
            remove_rows.append(i)
        elif(drugvar[protid]['len']>0):
            if split=='default':
                continue
            elif split=='closest':
                df = drugvar[protid]['var']
                max_row = df.loc[df['similarity'].idxmax()]
                if(max_row['similarity']>1500):
                    X_target.loc[protid]['Prot_Feat'] = max_row['feature'][0]
                    X_target.loc[protid]['SEQ'] = max_row['SEQ']
                    replaced_prots.append(protid)
                else:
                    remove_rows.append(i)
            elif split=='furthest':
                df = drugvar[protid]['var']
                max_row = df.loc[df['similarity'].idxmin()]
                if(max_row['similarity']<1000):
                    X_target.loc[protid]['Prot_Feat'] = max_row['feature'][0]
                    X_target.loc[protid]['SEQ'] = max_row['SEQ']
                    replaced_prots.append(protid)
                else:
                    remove_rows.append(i)
            elif split=='random':
                # Select random
                print("r", end="")
                df = drugvar[protid]['var']
                max_row = df.sample(n=1,random_state=1111)
                X_target.loc[protid]['Prot_Feat'] = max_row['feature'].values[0][0]
                X_target.loc[protid]['SEQ'] = max_row['SEQ']
                replaced_prots.append(protid)
        else:
            remove_rows.append(i)

    # indexes_to_remove = test_ind.iloc[remove_rows].index
    train_ind = train_ind.drop(remove_rows)

    print("replace_train proteins ", len(replaced_prots), "Unique ", len(set(replaced_prots)))
    print("replace_train Removed rows ", len(remove_rows))
    print("train_ind ", train_ind.shape)
    return X_target, train_ind


def cold_protein(X_target_org, train_ind_orig, test_ind_orig, pickle_file):
    # remove all with variants from train and place them in test
    # keep proteins with no varaint only in train 
    X_target = X_target_org.copy()
    train_ind = train_ind_orig.copy()
    test_ind = test_ind_orig.copy()

    with open(pickle_file, "rb") as f:
        drugvar = pickle.load(f)
    replaced_prots = []
    remove_train_rows = []
    for i,row in train_ind.iterrows():
        idx = row['Prot_ID']
        protid = idx
        if(protid not in drugvar):
            continue
        if(drugvar[protid]['len']>0):
            remove_train_rows.append(i)
            test_ind = test_ind.append({'Prot_ID':row['Prot_ID'], 'Drug_ID':row['Drug_ID'], 'label':row['label']}, ignore_index=True)

    train_ind = train_ind.drop(remove_train_rows)

    remove_rows = []
    for i,row in test_ind.iterrows():
        idx = row['Prot_ID']
        protid = idx
        if(protid not in drugvar):
            remove_rows.append(i)
            train_ind = train_ind.append({'Prot_ID':row['Prot_ID'], 'Drug_ID':row['Drug_ID'], 'label':row['label']}, ignore_index=True)
        if(drugvar[protid]['len']>0):
            continue
        else:
            remove_rows.append(i)
            train_ind = train_ind.append({'Prot_ID':row['Prot_ID'], 'Drug_ID':row['Drug_ID'], 'label':row['label']}, ignore_index=True)
    test_ind = test_ind.drop(remove_rows)

    return X_target, train_ind, test_ind

from process_data.test_logs.utils import generate_hashid, three_to_one, get_residues
import os
def get_modified_seq(seq,lig):
    hash = f"{generate_hashid(seq)}_{generate_hashid(lig)}"
    path = f"/data/istiaq/work/projects/dti/DTIVAR/process_data/test_logs/def_clos_fur_complex_drug/{hash}/{hash}_complex.pdb"
    if os.path.isfile(path):
        residues, indices = get_residues(path)
        seq2 = seq[:indices[0]] + seq[indices[-1]:]
        return 1,seq2
    else:
        return 0,seq