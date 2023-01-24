from sksurv.metrics import concordance_index_censored
import numpy as np
import torch

def evaluate(dataloader, model, device='cpu'):
    indicator, time, risk, labels, losses = [], [], [], [], []

    model = model.to(device)
    for batch in dataloader:
        y, delta, X, label = map(lambda x:x.to(device), batch)
        
        y_sort, indices = torch.sort(y,dim=0,descending=True)
        X_sort = X[indices]
        X_sort = torch.squeeze(X[indices],1)
        delta_sort = delta[indices]
        label_sort = label[indices]
        
        f_X = model(X_sort)
        f_X_exp = torch.exp(f_X) 
        risk_set = torch.log(torch.cumsum(f_X_exp,dim=0))
        loss = -torch.sum((f_X-risk_set)*delta_sort)

        indicator.append(delta_sort)
        time.append(y_sort)
        risk.append(f_X)
        labels.append(-label_sort)
        losses.append(loss.item())

    indicator = torch.cat(indicator,dim=0).bool().numpy().reshape(-1)
    time = torch.cat(time,dim=0).numpy().reshape(-1)
    risk = torch.cat(risk,dim=0).detach().numpy().reshape(-1)
    labels = torch.cat(labels,dim=0).numpy().reshape(-1)

    c_index = concordance_index_censored(indicator, time, risk)
    oracle_c_index = concordance_index_censored(indicator, labels, risk)

    return c_index[0], oracle_c_index[0], np.mean(losses)