import torch.utils.data as tud
import torch
from data import simulate_mnist_data
from utils import SurvDataSet, LeNet
from loss import batchRecombineLoss, bigSurvSGDLoss, coxCCLoss
from train import train_net


def flow(n,eta,prob_censoring,cfg):
    train, valid, test = simulate_mnist_data(n,eta=eta,prob_censoring=prob_censoring,seed=0)

    y_train, delta_train, X_train, label_train = train
    y_valid, delta_valid, X_valid, label_valid = valid
    y_test, delta_test, X_test, label_test = test

    trainDataset = SurvDataSet(y_train, delta_train, X_train, label_train)
    validDataset = SurvDataSet(y_valid, delta_valid, X_valid, label_valid)
    testDataset = SurvDataSet(y_test, delta_test, X_test, label_test)

    trainDataloader = tud.DataLoader(trainDataset, batch_size = cfg['batch_size'], shuffle=True)
    validDataloader = tud.DataLoader(validDataset, batch_size = cfg['batch_size'], shuffle=False)
    testDataloader = tud.DataLoader(testDataset, batch_size = cfg['batch_size'], shuffle=False)

    model = LeNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])

    if cfg['method'] == 'BatchRecombine':
        loss_fn = batchRecombineLoss
    elif cfg['method'] == 'BigSurvSGD':
        loss_fn = bigSurvSGDLoss
    elif cfg['method'] == 'CoxCC':
        loss_fn = coxCCLoss

    train_net(trainDataloader, validDataloader, testDataloader,model, optimizer, loss_fn, epoch=cfg['epoch'], device=cfg['device'], valid_round=cfg['valid_round'], valid_c_index=cfg['valid_metric'])

