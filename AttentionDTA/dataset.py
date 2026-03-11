import torch
from torch.utils.data import Dataset
import numpy as np
import MDAnalysis as mda, os
from utils import generate_hashid
three_to_one = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"
}

CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

CHARISOSMILEN = 64

CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}

CHARPROTLEN = 25

def label_smiles(line, smi_ch_ind, MAX_SMI_LEN=100):
    X = np.zeros(MAX_SMI_LEN,dtype=np.int64())
    for i, ch in enumerate(line[:MAX_SMI_LEN]):
        X[i] = smi_ch_ind[ch]
    return X


def label_sequence(line, smi_ch_ind, MAX_SEQ_LEN=1200):
    X = np.zeros(MAX_SEQ_LEN,np.int64())
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = smi_ch_ind[ch]
    return X

class CustomDataSet(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __getitem__(self, item):
        return self.pairs[item]

    def __len__(self):
        return len(self.pairs)
    
class CustomTestData(Dataset):
    def __init__(self, pairs, cut=0):
        self.pairs = pairs
        self.cut = cut
    def __getitem__(self, item):
        a,b,c,d,e = self.pairs[item].strip().split()
        # d = get_3_prime(d,percent=self.cut)
        d = get_modified_seq(d,c,percent=self.cut)
        # d = get_modified_seq_alpha(d,c,percent=self.cut)
        return " ".join([a,b,c,d,e])
    def __len__(self):
        return len(self.pairs)

def collate_fn(batch_data,max_d=100,max_p=1200):
    N = len(batch_data)
    compound_new = torch.zeros((N, max_d), dtype=torch.long)
    protein_new = torch.zeros((N, max_p), dtype=torch.long)
    labels_new = torch.zeros(N, dtype=torch.float)
    for i,pair in enumerate(batch_data):
        pair = pair.strip().split()
        compoundstr, proteinstr, label = pair[-3], pair[-2], pair[-1]
        compoundint = torch.from_numpy(label_smiles(compoundstr, CHARISOSMISET,max_d))
        compound_new[i] = compoundint
        proteinint = torch.from_numpy(label_sequence(proteinstr, CHARPROTSET,max_p))
        protein_new[i] = proteinint
        labels_new[i] = float(label)
    return (compound_new, protein_new, labels_new)



def get_3_prime(seq, percent=0.1):
    n = len(seq)
    percent = 1-percent
    return seq[:int(n*percent)]


def get_modified_seq(seq,lig,percent=0.1,cutoff_distance = 4.0):
    hash = f"{generate_hashid(seq)}_{generate_hashid(lig)}"
    path = f"/data/istiaq/work/projects/dti/DTIVAR/datasets/bindingDB/diffdock_custom_test/{hash}/{hash}_complex.pdb"
    # path = f"/data/istiaq/work/projects/dti/NEWTON_outputs/def_clos_fur_bind_out/{hash}/{hash}_complex.pdb"
    if not os.path.isfile(path):
        print("NOT FOUND2,",hash)
        return seq
    u = mda.Universe(path)
    ligand = u.select_atoms("resname UNL")
    binding_site_residues = u.select_atoms(f"protein and around {cutoff_distance} group ligand", ligand=ligand)
    residues = {(res.resname, res.resid) for res in binding_site_residues.residues}
    residues = sorted(residues, key=lambda x:x[1])
    ress = "".join([three_to_one[x[0]] for x in residues])
    indices = [y-1 for x,y in residues]

    ########### wholel part REPLACE with negative site #########################    
    # lighash = generate_hashid(lig)
    # if lighash in neg_binding_sites:
    #     neg = random.choice(neg_binding_sites[lighash])
    # else:
    #     neg = ""
    #     print("#$-$#")
    # seq2 = seq[:indices[0]] + neg + seq[indices[-1]:]
    ########### wholel part REPLACE with negative site #########################   

    ########### wholel part REPLACE with POSITIVE site #########################    
    # lighash = generate_hashid(lig)
    # if lighash in pos_binding_sites:
    #     pos = random.choice(pos_binding_sites[lighash])
    # else:
    #     pos = ""
    #     print("#$-$#")
    # seq2 = seq[:indices[0]] + pos + seq[indices[-1]:]
    ########### wholel part REPLACE with POSITIVE site #########################  
      

    ########### wholel part/ residue delete #########################    
    # seq2 = seq[:indices[0]] + seq[indices[-1]:]
    # seq2 = "".join([seq[i] for i in range(len(seq)) if i not in indices])
    ########### wholel part delete #########################    

    ########### percent delete #########################
    percentage = 1-percent
    k = int(((indices[-1]-indices[0])*percentage)//2)
    seq2 = seq[:indices[0]+k] + seq[indices[-1]-k:]
    ########### percent delete #########################
    # print("LEN ", len(seq2))
    # seq2 = seq
    if len(seq2)>len(seq):
        seq2 = seq
    return seq2




def get_modified_seq_alpha(seq,lig,percent=0.1,cutoff_distance = 4.0):
    hash = f"{generate_hashid(seq)}_{generate_hashid(lig)}"
    path = f"/data/istiaq/work/projects/dti/NEWTON_outputs/def_clos_fur_bind_out/{hash}/{hash}_complex.pdb"
    if not os.path.isfile(path):
        # print("NOT FOUND,",hash)
        return seq
    u = mda.Universe(path)
    ligand = u.select_atoms("resname LIG")
    binding_site_residues = u.select_atoms(f"protein and around {cutoff_distance} group ligand", ligand=ligand)
    residues = {(res.resname, res.resid) for res in binding_site_residues.residues}
    residues = sorted(residues, key=lambda x:x[1])
    ress = "".join([three_to_one[x[0]] for x in residues])
    indices = [y-1 for x,y in residues]

    ########### wholel part REPLACE with negative site #########################    
    # lighash = generate_hashid(lig)
    # if lighash in neg_binding_sites:
    #     neg = random.choice(neg_binding_sites[lighash])
    # else:
    #     neg = ""
    #     print("#$-$#")
    # seq2 = seq[:indices[0]] + neg + seq[indices[-1]:]
    ########### wholel part REPLACE with negative site #########################   


    ########### wholel part REPLACE with POSITIVE site #########################    
    # lighash = generate_hashid(lig)
    # if lighash in pos_binding_sites:
    #     pos = random.choice(pos_binding_sites[lighash])
    # else:
    #     pos = ""
    #     print("#$-$#")
    # seq2 = seq[:indices[0]] + pos + seq[indices[-1]:]
    ########### wholel part REPLACE with POSITIVE site #########################  
      

    ########### wholel part/ residue delete #########################    
    # seq2 = seq[:indices[0]] + seq[indices[-1]:]
    # seq2 = "".join([seq[i] for i in range(len(seq)) if i not in indices])
    ########### wholel part delete #########################    


    ########### percent keep from both side of binding site #########################
    percentage = 1-percent
    k = int(((indices[-1]-indices[0])*percentage)//2)
    seq2 = seq[:indices[0]+k] + seq[indices[-1]-k:]
    # print("LEN ", len(seq2))
    ########### percent delete #########################
    if len(seq2)>len(seq):
        seq2 = seq
    return seq2

