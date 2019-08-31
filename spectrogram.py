import librosa
from librosa.feature import melspectrogram
import numpy as np
import os

data_root = '/users/gpu/vaibhavj/vctk/VCTK-Corpus/wav48/'
save_root = './spec2/'
if not os.path.exists(save_root):
    os.makedirs(save_root)

os.system('ls '+data_root+' > s.txt')
fs = open('s.txt', 'r')

for s in fs:
    s = s[:-1]
    s_path = data_root+s+'/'
    os.system('ls '+s_path+' > a.txt')
    fa = open('a.txt', 'r')
    
    save_dir = save_root+s+'/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for a in fa:
        a = a[:-1]
        a_path = s_path+a
        # start processing
        y, sr = librosa.load(a_path, sr=16000)
        S = librosa.feature.melspectrogram(y, sr=16000, n_mels=80, fmin=90, fmax=7600, n_fft=1024, hop_length=256)
        S = 20 * np.log10(np.maximum(1e-5, S))
        S = S - 16
        S = np.clip((S + 100.0) / 100.0, 0, 1)
        np.save(save_dir+a[:-4], S.T)
        print("processed: "+save_dir+a[:-4], S.T.shape)
        
    fa.close()









