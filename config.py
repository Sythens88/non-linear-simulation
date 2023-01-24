def get_config(method,bs=32,lr=1e-4,epoch=20,device='cpu',valid_round=200,valid_metric='oracle'):    
    config = {
        'method':method,
        'batch_size':bs,
        'lr':lr,
        'epoch':epoch,
        'device':device,
        'valid_round':valid_round,
        'valid_metric':valid_metric
    }
    return config