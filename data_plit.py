import pandas as pd
import os 
import csv
from tqdm import tqdm
 
root = 'D:\\Big_Data\\SOCOFing\\Repaired\\Repaired-Hard_20'
dst = 'data/class_from_vae'
filename = 'Hard_20.csv'

f = open(os.path.join(dst, filename), 'w', newline='')
writer = csv.writer(f)
writer.writerow(["Image Name", "Number", "Gender", "Hand", "Finger", "Alteration"])

for filename in tqdm(os.listdir(root)):
    attr = filename.split('_')

    writer.writerow([filename, attr[0], attr[2], attr[3], attr[4], attr[6].split('.')[0]])

f.close()

