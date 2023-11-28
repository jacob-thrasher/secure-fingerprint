from data import *
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from network import CNNClassifier

root = 'D:\\Big_Data\\SOCOFing\\SOCOFing\\Real'
train, test = get_train_test_split_socofing(root)
dataset = SOCOFing(root, train)

dataloader = DataLoader(dataset, batch_size=32)
batch = next(iter(dataloader))
print(batch.size())

model = CNNClassifier(1, 500)

out = model(batch)
print(len(out[0]))
print(sum(out[0]))

# subject = dataset[0]
# print(subject)
# print(subject.size())
# plt.imshow(subject['img'], cmap='gray')
# plt.show()