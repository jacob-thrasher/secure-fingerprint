from data import *
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from network import CNNClassifier, VAE
import matplotlib.pyplot as plt

root = 'D:\\Big_Data\\SOCOFing\\SOCOFing\\Real'
train, test = get_train_test_split_socofing(root)
dataset = SOCOFing(root, train)

dataloader = DataLoader(dataset, batch_size=32)
batch = next(iter(dataloader))
print(batch.size())

model = VAE(num_hiddens=128, 
              num_residual_layers=2, 
              num_residual_hiddens=32,
              z_dim=64)

out = model(batch)
print(out[1].size())

plt.imshow(out[1][0].squeeze().detach().numpy(), cmap='gray')
plt.show()
# subject = dataset[0]
# print(subject)
# print(subject.size())
# plt.imshow(subject['img'], cmap='gray')
# plt.show()