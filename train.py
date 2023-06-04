# train
import os
import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader
from dataset import KuaishouDataset
from mestrics import xauc_score

def eval(model, eval_cfg): 
    model.zero_grad()
    batch_size  = eval_cfg["batch_size"]
    device      = torch.device(eval_cfg["device"] if torch.cuda.is_available() else "cpu")
    eval_data   = KuaishouDataset(eval_cfg["data_path"])
    eval_loader = DataLoader(dataset=eval_data, batch_size=batch_size, shuffle=False)
    
    all_preds  = []
    all_gtruth = []
    for data in eval_loader:
        inputs = [d.to(device) for d in data[:-1]]
        all_gtruth.append(data[-1].reshape(shape=[-1]))
        preds = model(inputs)
        all_preds.append(preds.reshape(shape=[-1]))

    all_preds  = torch.cat(all_preds, 0).cpu().detach().numpy()
    all_gtruth = torch.cat(all_gtruth, 0).cpu().detach().numpy()
    torch.cuda.empty_cache()
    return xauc_score(all_gtruth, all_preds)


def train(model, cfg):
    train_cfg    = cfg["train_cfg"]
    batch_size   = train_cfg["batch_size"]
    epoch_num    = train_cfg["epoch_num"]
    device       = torch.device(train_cfg["device"] if torch.cuda.is_available() else "cpu")
    train_data   = KuaishouDataset(train_cfg["data_path"])
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)
    loss_fn      = nn.BCELoss(reduce=False)
    optimizer1   = optim.Adam(model.backbone.parameters(), lr=train_cfg["lr"])
    optimizer2   = optim.Adam(model.discriminator.parameters(), lr=train_cfg["lr"])
    optimizer3   = optim.Adam(model.dfm.parameters(), lr=train_cfg["lr"])
    loss_fn      = nn.BCELoss(reduce=True, reduction='mean')

    print("device: ", device)
    global_step = 0
    model.to(device)

    for epoch in range(epoch_num): 
        for step, data in enumerate(train_loader):
            # train the discriminator
            optimizer2.zero_grad()
            data      = [d.to(device) for d in data]
            group_id  = data[-2].to(torch.int64).reshape(shape=[-1]).to(device)
            gtruth    = data[-1].reshape(shape=[-1]).to(device)
            real_labs = torch.nn.functional.one_hot(group_id, cfg['m_d']['num_class']).to(torch.float).to(device)
            r         = model.get_interest_representation(data)
            cls_preds = model.discriminator(r.data)
            loss_d    = loss_fn(cls_preds, real_labs)
            loss_d.backward(retain_graph=True)
            optimizer2.step()

            # train the adn
            optimizer1.zero_grad()
            optimizer3.zero_grad()
            fake_labs = torch.ones_like(real_labs) / float(cfg['m_d']['num_class'])
            cls_preds = model.discriminator(r)
            loss_g    = loss_fn(cls_preds, fake_labs)
            preds     = model(data)
            loss_w    = torch.nn.functional.huber_loss(preds, gtruth, reduction='mean', delta=0.5)
            loss = loss_w + cfg['alpha'] * loss_g
            loss.backward()
            optimizer1.step()
            optimizer3.step()

            global_step += 1
            if global_step % 1000 == 0:
                print('Epoch %d, Golbal_step: %d:Loss=%.4f' % (epoch+1, global_step, loss.item()))
        xauc = eval(model, cfg["eval_cfg"])
        torch.save(model.state_dict(), train_cfg["ckpt_path"] + "{}_{}_{:.6}.pkl".format(epoch+1, global_step, xauc))