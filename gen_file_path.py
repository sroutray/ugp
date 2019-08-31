import os

os.system("ls emb > f.txt")
f = open('f.txt')

for i in f:
	i = i[:-1]
	dir_command = "ls ./emb/"+i+"/ > f1.txt"
	os.system(dir_command)
	f1 = open('f1.txt')
	for j in f1:
		stri = "./emb/"+i+"/"+j[:-1]
		print(stri)