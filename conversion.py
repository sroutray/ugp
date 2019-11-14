import os
import pickle
import torch
import numpy as np
from math import ceil
from model_vc import Generator

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def pad_seq(x, base=32):
    len_out = int(base * ceil(float(x.shape[0])/base))
    len_pad = len_out - x.shape[0]
    assert len_pad >= 0
    return np.pad(x, ((0,len_pad),(0,0)), 'constant'), len_pad

if __name__ == "__main__":
    ## load data file
    data_root = '../test_files/'
    fh = open(os.path.join(data_root,'metadata.txt'), 'r')
    spks = fh.readlines()
    fh.close()
    spks = [s[:-1] for s in spks]
    specs = []
    embs = []
    for s in spks:
        emb = np.load(os.path.join(data_root,s+'.npy'))
        spec = np.load(os.path.join(data_root,s+'-feats.npy'))
        embs.append(emb)
        specs.append(spec)
    
    G = Generator(32,256,512,32).to(device)
    g_chkpt = torch.load('./weights/best_161238.pt')
    G.load_state_dict(g_chkpt['model'])

    spect_vc = []
    for i in range(0,len(spks)):
                
        x_org = specs[i]
        x_org, len_pad = pad_seq(x_org)
        uttr_org = torch.from_numpy(x_org[np.newaxis, :, :]).to(device)
        emb_org = torch.from_numpy(embs[i][np.newaxis, :]).to(device)
        ##################
        #if len_pad == 0:
        #    uttr_trg = uttr_org[0, :, :].cpu().numpy()
        #else:
        #    uttr_trg = uttr_org[0, :-len_pad, :].cpu().numpy() 
        #spect_vc.append( ('{}'.format(spks[i]), uttr_trg) )
        ####################

        
        for j in range(0,len(spks)):
                    
            emb_trg = torch.from_numpy(embs[j][np.newaxis, :]).to(device)
            
            with torch.no_grad():
                _, x_identic_psnt, _ = G(uttr_org, emb_org, emb_trg)
                
            if len_pad == 0:
                uttr_trg = x_identic_psnt[0, 0, :, :].cpu().numpy()
            else:
                uttr_trg = x_identic_psnt[0, 0, :-len_pad, :].cpu().numpy()
            
            spect_vc.append( ('{}x{}'.format(spks[i], spks[j]), uttr_trg) )
            print('{}x{}'.format(spks[i], spks[j]))

    with open('results.pkl', 'wb') as handle:
        pickle.dump(spect_vc, handle)
