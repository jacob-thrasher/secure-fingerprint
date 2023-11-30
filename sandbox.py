from data import *
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from network import CNNClassifier, VAE
import matplotlib.pyplot as plt

print(os.getcwd())
train_root = '/home/jacob/Documents/data/archive/SOCOFing/Altered/Altered-Hard'
gt_root = '/home/jacob/Documents/data/archive/SOCOFing/Real'
train_samples, test_samples = socofing_train_test_split_gen(train_root, test_size=32)

print(len(train_samples), len(test_samples))

train_dataset = SOCOFing_Gen(train_root, gt_root, train_samples, test_samples)

train, gt = train_dataset[0]
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(train, cmap='gray')
ax2.imshow(gt, cmap='gray')
fig.savefig('test.png')

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