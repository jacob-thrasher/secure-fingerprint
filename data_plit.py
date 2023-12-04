import pandas as pd
import os 
import csv
from tqdm import tqdm
 
root = '/home/jacob/Documents/data/archive/SOCOFing/Repaired/Hard'

f = open('data/class_test_from_vae_hard.csv', 'w', newline='')
writer = csv.writer(f)
writer.writerow(["Image Name", "Number", "Gender", "Hand", "Finger", "Alteration"])

for filename in tqdm(os.listdir(root)):
    attr = filename.split('_')

    writer.writerow([filename, attr[0], attr[2], attr[3], attr[4], attr[6].split('.')[0]])

f.close()

