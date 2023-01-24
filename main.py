from base import flow
from config import get_config
import logging
import argparse

def main(n,eta,method):
    logging.basicConfig(filename='logs/{}_{}_{}.log'.format(n,eta,method),filemode="w",format="%(message)s",level=logging.INFO)
    cfg = get_config(method)
    flow(n,eta,0.2,cfg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('n', type=int, default=50000)
    parser.add_argument('eta', type=int, default=1)
    parser.add_argument('method', type=int, default=1, help='1:BatchRecomb, 2:BigSurvSGD, 3:CoxCC')
    #parser.add_argument('time', type=int, default=10)
    args = parser.parse_args()

    n = int(args.n)
    eta = int(args.eta)
    if args.method == 1:
        method = 'BatchRecombine'
    elif args.method == 2:
        method = 'BigSurvSGD'
    elif args.method == 3:
        method = 'CoxCC'
    
    main(n,eta,method)