import torch
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve, confusion_matrix, precision_score, recall_score, auc
from torch.autograd import Variable
import numpy as np
def test(data_generator, model, device='cuda:1'):
    y_pred = []
    y_label = []
    model.eval()
    loss_accumulate = 0.0
    count = 0.0
    for i, (d, p, d_mask, p_mask, label) in enumerate(data_generator):
        score = model(d.long().to(device), p.long().to(device), d_mask.long().to(device), p_mask.long().to(device))
        
        m = torch.nn.Sigmoid()
        logits = torch.squeeze(m(score))
        loss_fct = torch.nn.BCELoss()            
        
        label = Variable(torch.from_numpy(np.array(label)).float()).to(device)

        loss = loss_fct(logits, label)
        
        loss_accumulate += loss
        count += 1
        
        logits = logits.detach().cpu().numpy()
        
        label_ids = label.to('cpu').numpy()
        y_label = y_label + label_ids.flatten().tolist()
        y_pred = y_pred + logits.flatten().tolist()
    auc = roc_auc_score(y_label, y_pred)
    av_pre = average_precision_score(y_label, y_pred)
    y_pred = [round(x) for x in y_pred]
    tn, fp, fn, tp = confusion_matrix(y_label, y_pred).ravel()
    f1 = f1_score(y_label, y_pred)
    return auc, av_pre, f1, tn, fp, fn, tp
