from data import *
import matplotlib.pyplot as plt

root = 'D:\\Big_Data\\SOCOFing\\SOCOFing\\Real'
train, test = get_train_test_split_socofing(root)
dataset = SOCOFing(root, train)

subject = dataset[0]
print(subject)
plt.imshow(subject['img'], cmap='gray')
plt.show()