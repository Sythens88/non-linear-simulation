import torch
import torch.nn as nn
import torch.utils.data as tud
import random


## Net
class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        self.lenet = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=16,kernel_size=5,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Flatten(),
            nn.Linear(32*5*5,120),
            nn.ReLU(),
            nn.Linear(120,60),
            nn.ReLU(),
            nn.Linear(60,1)
        )

    def forward(self, X):
        return self.lenet(X)

## Dataset

class SurvDataSet(tud.Dataset):
    def __init__(self, y, delta, X, label, sort=True):

        y = torch.Tensor(y)
        delta = torch.LongTensor(delta)
        X = torch.Tensor(X)
        label = torch.LongTensor(label)

        if sort:
            self.y, indices = torch.sort(y,dim=0,descending=False)
            self.X = X[indices]
            self.delta = delta[indices]
            self.label = label[indices]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.y[idx], self.delta[idx], self.X[idx], self.label[idx]



## batchRecombine

def getCaseIdx(bs):
    return {i:random.choice([j for j in range(i+1,bs)]) for i in range(bs-1)}

def getRecombineCase(bs):
    caseIdx = getCaseIdx(bs)
    case = []
    vis = set()
    for j in range(bs-1):
        if j in vis:
            continue
        idx = j
        group = [idx]
        while idx != bs-1:
            idx_next = caseIdx[idx]
            group.append(idx_next)
            if idx_next in vis:
                break
            else:
                vis.add(idx_next)
                idx = idx_next
        if len(group)>1:
            case.append(group)
    return case