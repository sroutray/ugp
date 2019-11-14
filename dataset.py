# we require a file emb list containing the links to emb files relative to the location of this file
import torch.utils.data as data
import torch
import numpy as np
from math import ceil
import os

def pad_seq(x, base=32, maxlen=None):
    if maxlen is None:
        maxlen = x.shape[0]
    len_out = int(base * ceil(float(maxlen)/base))
    len_pad = len_out - x.shape[0]
    assert len_pad >= 0
    return np.pad(x, ((0,len_pad),(0,0)), 'constant'), len_pad

def collate_fn(batch):
    input_lengths = [x[1].shape[0] for x in batch]
    maxlen = max(input_lengths)
    padded_batch = [pad_seq(x[1],base=32,maxlen=maxlen) for x in batch]

    e = torch.FloatTensor([x[0] for x in batch])
    x = torch.FloatTensor([x[0] for x in padded_batch])
    l = torch.IntTensor([x[1] for x in padded_batch])

    return e, x, l

class GetLoader(data.Dataset):
    def __init__(self, data_root, phase):
        f = open(os.path.join(data_root,phase+'.txt'), 'r')
        metadata_list = f.readlines()
        f.close()

        self.n_data = len(metadata_list)

        self.emb_paths = []
        self.spec_paths = []

        for line in metadata_list:
            line_list = line.split('|')
            spec_path = os.path.join(data_root, line_list[1])
            emb_path = os.path.join(data_root, line_list[0][0:4]+'_10avg.npy')
            self.emb_paths.append(emb_path)
            self.spec_paths.append(spec_path)

    def __getitem__(self, item):
        emb_path, spec_path = self.emb_paths[item], self.spec_paths[item]
        emb = np.load(emb_path)
        spec = np.load(spec_path)
        return emb, spec

    def __len__(self):
        return self.n_data


# data_root = '../debug_out/'
# batch_size = 2
# trainset = GetLoader(data_root)
# trainloader = torch.utils.data.DataLoader(
#     dataset=trainset,
#     batch_size=batch_size,
#     shuffle=True,
#     num_workers=4,
#     collate_fn=collate_fn)

# dataiter = iter(trainloader)
# e, x, l = dataiter.next()
# print(e.shape, x.shape, l)
