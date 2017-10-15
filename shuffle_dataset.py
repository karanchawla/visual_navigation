import sys
import random

csv_path = "/home/karan/Fall/research/visual_navigation/driving_log.csv"
fid = open(csv_path,'r')
li = fid.readlines()
fid.close()

random.shuffle(li)

fid = open(csv_path,'w')
fid.writelines(li)
fid.close()
print("Done shuffling your dataset")