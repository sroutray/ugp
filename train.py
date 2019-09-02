import time
import torch
from math import ceil

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import os
from torch.utils.data import DataLoader
from dataset import GetLoader

from model_vc import Generator


def train_model():
    if not os.path.exists('./weights/'):
        os.makedirs('./weights')

    if not os.path.exists('./logs/'):
        os.makedirs('./logs/')

    latest = './weights/latest.pt'
    log_file = './logs/train_logs.txt'
    
    #hyperparameters
    mu = 1.0
    lm = 1.0
    epochs = 10000

    device = 'cuda'
    G = Generator(32,256,512,32).to(device)

    loss_r = torch.nn.MSELoss()
    loss_r0 = torch.nn.MSELoss()
    loss_c = torch.nn.MSELoss()

    optimizer = optim.Adam(G.parameters())
    
    emb_list = "./emb_list1.txt"
    
    batch_size = 1
    trainset = GetLoader(emb_list)
    trainloader = torch.utils.data.DataLoader(
        dataset=trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4)
    
    te = time.time()
    for epoch in range(0, epochs):

        dataiter = iter(trainloader)
        mloss = 0
        tb = time.time()
        for i in range(len(trainloader)):
            loss = 0
            
            emb_org, x_org, pad_len = dataiter.next()
            emb_trg = emb_org
            
            # assume have uttr_org, emb_org, emb_trg
            x_org = x_org.to(device)
            emb_org = emb_org.to(device)
            emb_trg = emb_trg.to(device)

            xr, xr_psnt, code_x = G(x_org, emb_org, emb_trg)
            code_xr = G.encoder(xr_psnt, emb_org)
            code_xr = torch.cat(code_xr, dim=-1)

            loss = loss_r(xr_psnt, x_org) + mu*loss_r0(xr, x_org) + lm*loss_c(code_xr, code_x)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            mloss = (i*mloss + loss)/(i+1)
            
            print('Epoch: '+str(epoch+1)+'/'+str(epochs)+'   Minibatch: '+str(i+1)+'/'+str(len(trainloader))+
                  '   Loss: '+str(mloss.data.cpu().item())+'   Batch time(s): '+str(time.time()-tb)+
                  '   Epoch time(min): '+str((time.time()-te)/60))
            
            tb = time.time()
            

        # Create checkpoint
        with open(log_file, 'a') as fh:
            fh.write(str(i)+' '+str(mloss.item())+'\n')

        chkpt = {'epoch': epoch,
                 'loss': mloss.item(),
                 'model': G.state_dict(),
                 'optimizer': optimizer.state_dict()}
        torch.save(chkpt, latest)
        
        te = time.time()
        
if __name__=='__main__':
#     if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--epochs', type=int, default=68, help='number of epochs')
#     parser.add_argument('--batch-size', type=int, default=8, help='batch size')
#     parser.add_argument('--accumulate', type=int, default=8, help='number of batches to accumulate before optimizing')
#     parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='cfg file path')
#     parser.add_argument('--data-cfg', type=str, default='data/coco_64img.data', help='coco.data file path')
#     parser.add_argument('--resume', action='store_true', help='resume training flag')
#     opt = parser.parse_args()
#     print(opt)
    train_model()
