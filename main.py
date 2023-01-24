from base import flow
from config import get_config
import logging
import argparse

def main(n,eta,method,bs,lr,epoch,device,valid_round,valid_metric):
    logging.basicConfig(filename='logs/{}_{}_{}.log'.format(n,eta,method),filemode="w",format="%(message)s",level=logging.INFO)
    cfg = get_config(method,bs,lr,epoch,device,valid_round,valid_metric)
    flow(n,eta,0.2,cfg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('n', type=int, default=50000)
    parser.add_argument('eta', type=int, default=1)
    parser.add_argument('method', type=int, default=1, help='1:BatchRecomb, 2:BigSurvSGD, 3:CoxCC')
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--valid-round', type=int, default=200)
    parser.add_argument('--valid-metric', type=int, default=1, help='1:oracle,2:else')
    parser.add_argument('--device', type=int, default=1, help='1:cpu, 2:cuda')
    #parser.add_argument('--time', type=int, default=10)
    args = parser.parse_args()

    n = int(args.n)
    eta = int(args.eta)
    bs = int(args.bs)
    lr = float(args.lr)
    epoch = int(args.epoch)
    valid_round = int(args.valid_round)
    if int(args.method) == 1:
        method = 'BatchRecombine'
    elif int(args.method) == 2:
        method = 'BigSurvSGD'
    elif int(args.method) == 3:
        method = 'CoxCC'
    if int(args.valid_metric) == 1:
        valid_metric = 'oracle'
    else:
        valid_metric = 'c-index'
    if int(args.device) == 1:
        device = 'cpu'
    else:
        device = 'cuda'
        
    main(n,eta,method,bs,lr,epoch,device,valid_round,valid_metric)