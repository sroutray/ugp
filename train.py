import argparse
import sys
import time
from datetime import datetime
from math import ceil

from tensorboardX import SummaryWriter

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import os
from torch.utils.data import DataLoader
from dataset import GetLoader, pad_seq, collate_fn

from model_vc import Generator

start_epoch = 0
global_train_step = 0
global_test_step = 0
best_loss = 0
curr_epoch = 0
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class MaskedMSELoss(torch.nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, target, inp, mask):
        loss = torch.sum((inp*mask - target*mask)**2) / mask.sum()
        return loss

def get_mask(x, l):
    m = np.ones(x.shape)
    zs = []
    for i in range(len(l)):
        if l[i].item() != 0:
            zs.append(np.zeros(m[i,-l[i].item():,:].shape))
        else:
            zs.append(None)
    
    for i in range(len(l)):
        if l[i].item() != 0:
            m[i,-l[i].item():,:] = zs[i]

    return torch.FloatTensor(m)


def train_model(G, optimizer, trainloader, testloader,
                epochs, writer):
    global start_epoch, global_train_step, global_test_step, best_loss, curr_epoch, device
        
    #hyperparameters
    mu = 1.0
    lm = 1.0

    loss_r = MaskedMSELoss().to(device)
    loss_r0 = MaskedMSELoss().to(device)
    loss_c = torch.nn.L1Loss().to(device)
    
    te = time.time()
    for epoch in range(start_epoch, epochs):
        curr_epoch = epoch

        trainiter = iter(trainloader)
        runningloss = 0
        tb = time.time()
        G.train()
        for i in range(len(trainloader)):
            loss = 0
            
            emb_org, x_org, pad_len = trainiter.next()
            emb_trg = emb_org
            mask = get_mask(x_org, pad_len)

            # assume have uttr_org, emb_org, emb_trg
            x_org = x_org.to(device)
            emb_org = emb_org.to(device)
            emb_trg = emb_trg.to(device)
            mask = mask.to(device)

            xr, xr_psnt, code_x = G(x_org, emb_org, emb_trg)
            code_xr = G(xr_psnt, emb_org)

            xr = xr[:,0,:,:]
            xr_psnt = xr_psnt[:,0,:,:]
            loss = loss_r(xr_psnt, x_org, mask) + mu*loss_r0(xr, x_org, mask) + lm*loss_c(code_xr, code_x)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            runningloss += loss
            
            print('Train Epoch: '+str(epoch+1)+'/'+str(epochs)+'   Minibatch: '+str(i+1)+'/'+str(len(trainloader))+
                  '   Loss: '+str(loss.data.cpu().item())+'   Batch time(s): '+str(time.time()-tb)+
                  '   Epoch time(min): '+str((time.time()-te)/60))
            
            tb = time.time()
            writer.add_scalar('Batch Train Loss', loss, global_train_step)
            global_train_step += 1

        testiter = iter(testloader)
        runningloss_test = 0
        G.eval()
        with torch.no_grad():
            for i in range(len(testloader)):
                loss = 0

                emb_org, x_org, pad_len = testiter.next()
                emb_trg = emb_org
                mask = get_mask(x_org, pad_len)

                # assume have uttr_org, emb_org, emb_trg
                x_org = x_org.to(device)
                emb_org = emb_org.to(device)
                emb_trg = emb_trg.to(device)
                mask = mask.to(device)

                xr, xr_psnt, code_x = G(x_org, emb_org, emb_trg)
                code_xr = G(xr_psnt, emb_org, c_trg=None)
                
                xr = xr[:,0,:,:]
                xr_psnt = xr_psnt[:,0,:,:]
                loss = loss_r(xr_psnt, x_org, mask) + mu*loss_r0(xr, x_org, mask) + lm*loss_c(code_xr, code_x)
                
                runningloss_test += loss

                print('Test Epoch: '+str(epoch+1)+'/'+str(epochs)+'   Minibatch: '+str(i+1)+'/'+str(len(testloader))+
                        '   Loss: '+str(loss.data.cpu().item())+'   Batch time(s): '+str(time.time()-tb)+
                        '   Epoch time(min): '+str((time.time()-te)/60))

                tb = time.time()
                writer.add_scalar('Batch Test Loss', loss, global_test_step)
                global_test_step += 1
        
        
        # Create checkpoint
        writer.add_scalar('Epoch Train Loss', runningloss/len(trainloader), epoch)
        writer.add_scalar('Epoch Test Loss', runningloss_test/len(testloader), epoch)

        if best_loss == 0:
            best_loss = runningloss/len(trainloader)
        elif runningloss/len(trainloader) < best_loss:
            best_loss = runningloss/len(trainloader)
            chkpt = {'epoch': epoch,
                    'model': G.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'train_step': global_train_step,
                    'test_step': global_test_step,
                    'best_loss': best_loss}
            torch.save(chkpt, './weights/best_'+str(global_train_step)+'.pt')
        
        chkpt = {'epoch': epoch,
                 'model': G.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'train_step': global_train_step,
                 'test_step': global_test_step,
                 'best_loss': best_loss}
        torch.save(chkpt, './weights/chkpt_'+str(global_train_step)+'.pt')   
        te = time.time()
        
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--log_event_path', type=str, default=None, help='tensorboard log path')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='restore from a checkpoint')
    opt = parser.parse_args()
    
    if not os.path.exists('./weights/'):
        os.makedirs('./weights')

    if not os.path.exists('./log/'):
        os.makedirs('./log/')
    
    train_data_root = '../data/'
    batch_size = opt.batch_size
    trainset = GetLoader(train_data_root, 'trainset')
    trainloader = torch.utils.data.DataLoader(
        dataset=trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn)
    
    test_data_root = '../data/'
    testset = GetLoader(test_data_root, 'testset')
    testloader = torch.utils.data.DataLoader(
        dataset=testset,
        batch_size=2,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn)

    G = Generator(32,256,512,32).to(device)

    optimizer = optim.Adam(G.parameters())

    #global start_epoch, global_train_step, global_test_step, best_loss, curr_epoch
    if opt.checkpoint_path is not None:
        chkpt = torch.load(opt.checkpoint_path)
        G.load_state_dict(chkpt['model'])
        optimizer.load_state_dict(chkpt['optimizer'])
        start_epoch = chkpt['epoch'] + 1
        curr_epoch = chkpt['epoch'] + 1
        global_train_step = chkpt['train_step']
        global_test_step = chkpt['test_step']
        best_loss = chkpt['best_loss']


    # Setup summary writer for tensorboard
    log_event_path = opt.log_event_path
    if log_event_path is None:
        log_event_path = "log/run-test" + str(datetime.now()).replace(" ", "_")
    print("TensorBoard event log path: {}".format(log_event_path))
    writer = SummaryWriter(log_dir=log_event_path)
    
    try:
        train_model(G, optimizer, trainloader, testloader,
                    opt.epochs, writer)
    except KeyboardInterrupt:
        print('Interrupted')
        pass
    finally:
        chkpt = {'epoch': curr_epoch,
                 'model': G.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'train_step': global_train_step,
                 'test_step': global_test_step,
                 'best_loss': best_loss}
        torch.save(chkpt, './weights/chkpt_'+str(global_train_step)+'.pt')

    print("Finished")

    sys.exit(0)
