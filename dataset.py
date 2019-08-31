# we require a file emb list containing the links to emb files relative to the location of this file
import torch.utils.data as data
import torch
import numpy as np
from math import ceil

def pad_seq(x, base=32):
    len_out = int(base * ceil(float(x.shape[0])/base))
    len_pad = len_out - x.shape[0]
    assert len_pad >= 0
    return np.pad(x, ((0,len_pad),(0,0)), 'constant'), len_pad

class GetLoader(data.Dataset):
    def __init__(self, emb_list):
        f = open(emb_list, 'r')
        emb_path_list = f.readlines()
        f.close()

        self.n_data = len(emb_path_list)

        self.emb_paths = []
        self.spec_paths = []

        for emb_path in emb_path_list:
            self.emb_paths.append(emb_path[:-1])
            spec_path = "./spec2/"+emb_path.split('/')[-2]+'/'+emb_path.split('/')[-1][:-1]
            self.spec_paths.append(spec_path)

    def __getitem__(self, item):
        emb_path, spec_path = self.emb_paths[item], self.spec_paths[item]
        emb = np.load(emb_path)
        spec = np.load(spec_path)
        
        spec, pad_len = pad_seq(spec)

        return emb, spec, pad_len

    def __len__(self):
        return self.n_data

# emb_list = "./emb_list.txt"
# batch_size = 1
# trainset = GetLoader(emb_list)
# trainloader = torch.utils.data.DataLoader(
#     dataset=trainset,
#     batch_size=batch_size,
#     shuffle=True,
#     num_workers=4)
    
# dataiter = iter(trainloader)
# for i in range(1000):
#     emb,spec,emb_path = dataiter.next()
#     print(emb.shape,spec.shape,emb_path)

