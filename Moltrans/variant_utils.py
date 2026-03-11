import numpy as np
import pandas as pd
import sys,ast 
import pickle

def replace_test(df_test, pickle_file, split, modify=False):
    print("TEEEEEEEEEEEEEEEEEESSSSSSSSSSSSSSSSSSSSTTTTTTTTTTTTTTT")
    with open(pickle_file, "rb") as f:
        drugvar = pickle.load(f)
    replaced_prots = []
    remove_rows = []
    for i,row in df_test.iterrows():
        seq = row['Target Sequence']
        if(seq not in drugvar):
            remove_rows.append(i)
        else: #(drugvar[protid]['len']>0):
            if split=='default':
                continue
            elif split=='closest':
                df = drugvar[seq]['var']
                max_row = df.loc[df['similarity'].idxmax()]
                if(max_row['similarity']>1500):
                    df_test.loc[i,'Target Sequence'] = max_row['SEQ']
                    replaced_prots.append(seq)
                else:
                    remove_rows.append(i)
            elif split=='furthest':
                df = drugvar[seq]['var']
                max_row = df.loc[df['similarity'].idxmin()]
                if(max_row['similarity']<1000):
                    df_test.loc[i,'Target Sequence'] = max_row['SEQ']
                    replaced_prots.append(seq)
                else:
                    remove_rows.append(i)
            elif split=='random':
                df = drugvar[seq]['var']
                max_row = df.sample(n=1,random_state=1111)
                df_test.loc[i,'Target Sequence'] = max_row['SEQ'].values[0]
                replaced_prots.append(seq)

    df_test = df_test.drop(remove_rows)
    print(df_test.head())
    df_test['Target Sequence'] = df_test.apply(lambda x: get_modified_seq(x['Target Sequence'],x['SMILES']),axis=1)
    print("Replaced proteins ", len(replaced_prots), "Unique ", len(set(replaced_prots)))
    print("Removed rows ", len(remove_rows))
    return df_test

def replace_train(df_train, pickle_file, split):
    with open(pickle_file, "rb") as f:
        drugvar = pickle.load(f)
    replaced_prots = []
    remove_rows = []
    for i,row in df_train.iterrows():
        seq = row['Target Sequence']
        if(seq not in drugvar):
            continue
            remove_rows.append(i)
        else: #(drugvar[protid]['len']>0):
            if split=='default':
                continue
            elif split=='closest':
                df = drugvar[seq]['var']
                max_row = df.loc[df['similarity'].idxmax()]
                if(max_row['similarity']>1500):
                    df_train.loc[i,'Target Sequence'] = max_row['SEQ']
                    replaced_prots.append(seq)
                else:
                    remove_rows.append(i)
            elif split=='furthest':
                df = drugvar[seq]['var']
                max_row = df.loc[df['similarity'].idxmin()]
                if(max_row['similarity']<1000):
                    df_train.loc[i,'Target Sequence'] = max_row['SEQ']
                    replaced_prots.append(seq)
                else:
                    remove_rows.append(i)
            elif split=='random':
                # Select random
                print("r", end="")
                df = drugvar[seq]['var']
                max_row = df.sample(n=1,random_state=1111)
                df_train.loc[i,'Target Sequence'] = max_row['SEQ']
                replaced_prots.append(seq)


    # df_train = df_train.drop(remove_rows)
    print("Train Replaced proteins ", len(replaced_prots), "Unique ", len(set(replaced_prots)))
    print("Train Removed rows ", len(remove_rows))
    return df_train


def cold_protein(df_train, df_test, pickle_file, split):
    orig_train = df_train.copy()
    orig_test = df_test.copy()
    with open(pickle_file, "rb") as f:
        drugvar = pickle.load(f)

    replaced_prots = []
    remove_train_rows = []
    for i,row in df_train.iterrows():
        seq = row['Target Sequence']
        if(seq not in drugvar):
            continue
        if(drugvar[seq]['len']>0):
            remove_train_rows.append(i)
            df_test = df_test.append({'Label':row['Label'], 'SMILES':row['SMILES'], 'Target Sequence':row['Target Sequence']}, ignore_index=True)

    df_train = df_train.drop(remove_train_rows)

    remove_rows = []
    for i,row in df_test.iterrows():
        seq = row['Target Sequence']
        if(seq not in drugvar):
            remove_rows.append(i)
            df_train = df_train.append({'Label':row['Label'], 'SMILES':row['SMILES'], 'Target Sequence':row['Target Sequence']}, ignore_index=True)
            continue
        if(drugvar[seq]['len']>0):
            continue
        else:
            remove_rows.append(i)
            train_ind = train_ind.append({'Label':row['Label'], 'SMILES':row['SMILES'], 'Target Sequence':row['Target Sequence']}, ignore_index=True)
    df_test = df_test.drop(remove_rows)

    return df_train, df_test

from mdautils import generate_hashid, three_to_one, get_residues
import os
def get_modified_seq(seq,lig):
    hash = f"{generate_hashid(seq)}_{generate_hashid(lig)}"
    path = f"/data/istiaq/work/projects/dti/DTIVAR/process_data/test_logs/def_clos_fur_complex_bind/{hash}/{hash}_complex.pdb"
    if os.path.isfile(path):
        residues, indices = get_residues(path)
        seq2 = seq[:indices[0]] + seq[indices[-1]:]
        print("MODDDDDDDDDDDDDDDDDDDDDDD")
        return seq2
    else:
        return seq