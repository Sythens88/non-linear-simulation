import torchvision
import numpy as np
import random

def simulate(data,prob_censoring=0.2,eta=1,seed=0):
    np.random.seed(seed)

    X = data.data.numpy()
    label = data.targets.numpy()
    n = len(X)
    mu = np.exp(-label*eta).reshape(-1,1)
    y = np.random.exponential(mu, (n,1))
    delta = np.random.binomial(n=1, p=1-prob_censoring,size=(n,1))

    return y, delta, X, label



def simulate_mnist_data(n,eta=1,prob_censoring=0.2,seed=0):
    train_data = torchvision.datasets.MNIST(root='./mnist/',train=True,download=False)
    test_data = torchvision.datasets.MNIST(root='./mnist/',train=False,download=False)

    ## get train and test
    train = simulate(train_data,prob_censoring=prob_censoring,eta=eta,seed=seed)
    test = simulate(test_data,prob_censoring=prob_censoring,eta=eta,seed=seed)

    ## split trainset to train and valid
    y_train, delta_train, X_train, label_train = train
    idx = [i for i in range(len(y_train))]
    random.seed(0)
    random.shuffle(idx)
    train_idx, valid_idx = idx[:50000], idx[50000:]
    if n<=50000:
        train_idx = train_idx[:n]

    ## warp train and valid
    y_valid, delta_valid, X_valid, label_valid = y_train[valid_idx], delta_train[valid_idx], X_train[valid_idx], label_train[valid_idx]
    y_train, delta_train, X_train, label_train = y_train[train_idx], delta_train[train_idx], X_train[train_idx], label_train[train_idx]
   
    train = (y_train, delta_train, X_train, label_train)
    valid = (y_valid, delta_valid, X_valid, label_valid)

    return train, valid, test
    
   

# train = simulate_mnist_data(train_data)
# test = simulate_mnist_data(test_data)


