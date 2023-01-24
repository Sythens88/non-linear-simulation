from copy import deepcopy
import torch
import logging
from evaluate import evaluate

def train_net(trainDataloader, validDataloader, testDataloader,\
          model, optimizer, compute_loss, \
          epoch=15, device='cpu', valid_round=100, valid_c_index='oracle'):
    best_model = deepcopy(model)
    model, best_model = model.to(device), best_model.to(device)
    max_c_index = float('-inf')

    for e in range(epoch):

        for idx, batch in enumerate(trainDataloader):
            y, delta, X, label = map(lambda x:x.to(device), batch)

            loss = compute_loss(y, delta, X, model)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx % valid_round==0:
                c_index, oracle_c_index, val_loss = evaluate(validDataloader, model, device=device)
                metric = oracle_c_index if valid_c_index=='oracle' else c_index
                logging.info('{},{},{},{},{}'.format(e, idx, c_index, oracle_c_index, val_loss))
                print('Epoch: {}, Iter: {}, C index:{:.4f}, Oracle C index: {:.4f}, Loss: {:.4f}'.format(e, idx, c_index, oracle_c_index, val_loss))
                if metric>=max_c_index:
                    best_model.load_state_dict(model.state_dict())
                    max_c_index = metric
                    print('Save new best model, current metric:{:.5f}'.format(metric))

        c_index, oracle_c_index, val_loss = evaluate(validDataloader, model, device=device)
        print('End of epoch: {}, C index:{:.4f}, Oracle C index: {:.4f}, Loss: {:.4f}'.format(e, c_index, oracle_c_index, val_loss))
    
    print('End of training. Begin testing.')

    c_index, oracle_c_index, test_loss = evaluate(testDataloader, model, device=device) 

    logging.info('Test,{},{},{}'.format(c_index, oracle_c_index, test_loss))
    print('Testset performance. C index:{:.4f}, Oracle C index: {:.4f}, Loss: {:.4f}'.format(c_index, oracle_c_index, test_loss)) 






