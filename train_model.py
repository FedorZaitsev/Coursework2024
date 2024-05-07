#!/usr/bin/env python3

import sys
import os
import albumentations as A
import torch
import gc
import torch.nn as nn

from classification import set_rng, seed_worker, parse_config, clf_train, get_loaders, static_quantize, get_model

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('Please enter config path and train/valid dataset directories')
        exit(0)
    cfg = parse_config(sys.argv[1])
    train_dir = sys.argv[2]
    valid_dir = None
    wandb_key = None
    wandb_proj_name = None
    if len(sys.argv) >= 4:
        valid_dir = sys.argv[3]
    if len(sys.argv) >= 6:
        wandb_key = sys.argv[4]
        wandb_proj_name = sys.argv[5]
    

    gen = set_rng(cfg['seed'])
    train_loader, valid_loader, classes = get_loaders(train_dir=train_dir, valid_dir=valid_dir, 
                                                      aug_cfg=cfg['aug_cfg'], train_ratio=cfg['train_ratio'], 
                                                      worker_init_fn=seed_worker, generator=gen)

    torch.cuda.empty_cache()
    gc.collect()

    model = cfg['model']['instance'](**cfg['model']['parameters'], num_classes=len(classes))
    num_epochs = cfg['num_epochs']
    optimizer = cfg['optimizer']['instance'](model.parameters(), **cfg['optimizer']['parameters'])
    scheduler = cfg['scheduler']['instance'](optimizer, **cfg['scheduler']['parameters'])
    loss_fn = cfg['loss_fn']['instance'](**cfg['loss_fn']['parameters'])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Current device is:', device)
    model.to(device)

    res = {}
    model_info = {}
    model_info['inference_info'] = {}
    model_info['model_type'] = cfg['model_type']
    model_info['inference_info']['classes'] = classes
    model_info['inference_info']['valid_transforms'] = {}
    if len(cfg['aug_cfg']['valid_transforms']) > 0:
        model_info['inference_info']['valid_transforms'] = cfg['aug_cfg']['valid_transforms'][0].to_dict()
    res['log'] = {}
    res['log']['training_stat'] = clf_train(model=model, num_epochs=num_epochs, title=cfg['title'], train_loader=train_loader,
         valid_loader=valid_loader, optimizer=optimizer, loss_fn=loss_fn, scheduler=scheduler, 
         wandb_log=wandb_key, key=wandb_key, proj_name=wandb_proj_name, verbose=True)
    
    torch.save(model.state_dict(), cfg['model_save_path'])
    model_info['state_dict'] = cfg['model_save_path']
    models = {cfg['title']: model_info}
    if cfg['quantize']:
        q_model = static_quantize(model=model, loader=train_loader, loss_fn=loss_fn)
        torch.save(q_model.state_dict(), cfg['q_model_save_path'])
        if len(valid_loader) == 0:
            accuracy = 0
        else:
            total = 0
            correct = 0
            for x, y in valid_loader:                   
                x = x.to('cpu')        
                y = y.type(torch.LongTensor)
                y = y.to('cpu')
                output = q_model(x)
                output.to('cpu')
                _, y_pred = torch.max(output, 1)
                total += y.size(0)
                correct += (y_pred == y).sum().item()
            accuracy = correct / total
        res['log']['quantized_acc'] = accuracy

        q_model_info = {}
        q_model_info['inference_info'] = model_info['inference_info']
        q_model_info['state_dict'] = cfg['q_model_save_path']
        q_model_info['model_type'] = 'Quantized' + cfg['model_type']
        models['Quantized' + cfg['title']] = q_model_info


    

    import json

    with open(cfg['title'] + '_log.json', 'w') as f:
        json.dump(res, f)

    current_models = None
    with open('models/config.json', 'r') as f:
        current_models = json.load(f)

    for key, value in models.items():
        current_models[key] = value
    
    with open('models/config.json', 'w') as f:
        json.dump(current_models, f)

    exit(0)
    
        

    
