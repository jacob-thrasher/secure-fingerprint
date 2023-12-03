from data import *
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from network import CNNClassifier, VAE
import matplotlib.pyplot as plt
import pandas as pd
import csv
from tqdm import tqdm

data_root = 'D:\\Big_Data\\SOCOFing\\SOCOFing\\Altered'
gt_root = 'D:\\Big_Data\\SOCOFing\\SOCOFing\\Real'

dataset = SOCOFing_Gen(data_root, gt_root, 'data/vae_train.csv', drop_difficulty=['Medium', 'Hard'])

df = dataset.df
df.to_csv('test.csv')

train, test = dataset[0]

print(train.size(), test.size())


# print(os.getcwd())
# train_root = '/home/jacob/Documents/data/archive/SOCOFing/Altered/Altered-Hard'
# gt_root = '/home/jacob/Documents/data/archive/SOCOFing/Real'
# train_samples, test_samples = socofing_train_test_split_gen(train_root, test_size=32)

# print(len(train_samples), len(test_samples))

# train_dataset = SOCOFing_Gen(train_root, gt_root, train_samples, test_samples)

# train, gt = train_dataset[0]
# fig, (ax1, ax2) = plt.subplots(1, 2)
# ax1.imshow(train, cmap='gray')
# ax2.imshow(gt, cmap='gray')
# fig.savefig('test.png')

# dataloader = DataLoader(dataset, batch_size=32)
# batch = next(iter(dataloader))
# print(batch.size())

# model = VAE(num_hiddens=128, 
#               num_residual_layers=2, 
#               num_residual_hiddens=32,
#               z_dim=64)

# out = model(batch)
# print(out[1].size())

# plt.imshow(out[1][0].squeeze().detach().numpy(), cmap='gray')
# plt.show()
# subject = dataset[0]
# print(subject)
# print(subject.size())
# plt.imshow(subject['img'], cmap='gray')
# plt.show()





# VAE train/test split code

# df = pd.read_csv('data\\test_data.csv')  
# train_subjects = list(df['Image Name'])

# diff = ['Easy', 'Medium', 'Hard']
# root = 'D:\\Big_Data\\SOCOFing\\SOCOFing\\Altered'

# f = open('data\\vae_test.csv', 'w', newline='')
# writer = csv.writer(f)
# writer.writerow(["Image Name", "Number", "Gender", "Hand", "Finger", "Alteration", "Difficulty"])

# for d in diff:
#     data_path = os.path.join(root, f'Altered-{d}')
#     for filename in tqdm(os.listdir(data_path)):
#         attr = filename.split('_')
#         gt_filename = attr[0] + '__' + attr[2] + '_' + attr[3] + '_' + attr[4] + '_finger.BMP'

#         if gt_filename in train_subjects:
#             writer.writerow([filename, attr[0], attr[2], attr[3], attr[4], attr[6].split('.')[0], d])

# f.close()