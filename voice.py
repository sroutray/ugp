import os
import torch
import librosa
import pickle
from synthesis import build_model
from synthesis import wavegen

spect_vc = pickle.load(open('results.pkl', 'rb'))
device = torch.device("cuda")
model = build_model().to(device)
checkpoint = torch.load("../wavenet_vocoder_legacy_finetune/checkpoints/checkpoint_step000020000_ema.pth")
model.load_state_dict(checkpoint["state_dict"])
out_dir = '../out_files-4'
os.makedirs(out_dir,exist_ok=True)

for spect in spect_vc:
    name = spect[0]
    f_path = os.path.join(out_dir,name+'.wav')
    c = spect[1]
    print(name)
    waveform = wavegen(model, c=c)   
    librosa.output.write_wav(f_path, waveform, sr=16000)
