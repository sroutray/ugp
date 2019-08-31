import os
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import numpy as np

os.system("ls VCTK-Corpus/wav48 > f.txt")
f = open("f.txt")
for i in f:
	i = i[:-1]
	st = "ls VCTK-Corpus/wav48/"+i+" > f1.txt"
	os.system(st)
	direc = "mkdir emb/"+i
	os.system(direc)
	fi = open("f1.txt")
	for j in fi:
		j = j[:-1]
		print(i,j)
		stri = "./VCTK-Corpus/wav48/"+i+"/"+j
		fpath = Path(stri)
		wav = preprocess_wav(fpath)
		encoder = VoiceEncoder()
		embed = encoder.embed_utterance(wav)
		j = j[:-4]
		out_file = "emb/"+i+"/"+j
		np.save(out_file,embed)
# np.set_printoptions(precision=3, suppress=True)
# print(embed)

