import torch
import random
from utils import getRecombineCase

def batchRecombineLoss(y, delta, X, model):
    y_sort, indices = torch.sort(y,dim=0,descending=False)
    X_sort = torch.squeeze(X[indices],1)
    X_sort_net = model(X_sort)
    delta_sort = delta[indices]

    bs = len(y)
    caseList = getRecombineCase(bs)

    loss = 0
    for case in caseList:
        case = case[::-1]
        X_net_case, delta_case = X_sort_net[case], delta_sort[case]
        X_net_exp_case = torch.exp(X_net_case)
        risk_set = torch.log(torch.cumsum(X_net_exp_case,dim=0))
        loss += -torch.sum((X_net_case-risk_set)*delta_case)/len(case)
    loss = loss/len(caseList)

    return loss

def bigSurvSGDLoss(y, delta, X, model):
    y_sort, indices = torch.sort(y,dim=0,descending=True)
    X_sort = torch.squeeze(X[indices],1)
    delta_sort = delta[indices]

    f_X = model(X_sort)
    f_X_exp = torch.exp(f_X) 
    risk_set = torch.log(torch.cumsum(f_X_exp,dim=0))
    loss = -torch.sum((f_X-risk_set)*delta_sort)

    return loss

def coxCCLoss(y, delta, X, model):
    y_sort, indices = torch.sort(y,dim=0,descending=False)
    X_sort = torch.squeeze(X[indices],1)
    delta_sort = delta[indices]

    bs = len(y)
    caseControl = {i:random.choice([j for j in range(i,bs)]) for i in range(bs-1)}
    indIdx = list(caseControl.keys())
    controlIdx = list(caseControl.values())

    X_sort_net = model(X_sort)
    ind_net = X_sort_net[indIdx]
    control_net = X_sort_net[controlIdx]

    loss = torch.sum(delta_sort[:-1]*torch.log(1+torch.exp(control_net - ind_net)))
    
    return loss