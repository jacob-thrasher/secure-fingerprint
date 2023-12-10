import torch
import os
from torch.utils.data import DataLoader
from data import SOCOFing_class
from network import SimpleCNN

torch.manual_seed(69)

data_root = 'D:\\Big_Data\\SOCOFing'

train_dataset = SOCOFing_class(csv_file='data/base/train_set_1.csv', root_dir=os.path.join(data_root, 'Real'))
val_dataset = SOCOFing_class(csv_file='data/base/test_set_1.csv', root_dir=os.path.join(data_root, 'Real'))
# test_dataset = SOCOFing_class(csv_file= 'data/base/test_set_2.csv', root_dir=os.path.join(data_root, 'Real'))
# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = SimpleCNN()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, verbose=True)

if torch.cuda.is_available(): device = 'cuda'
else: device = 'cpu'

print(f'Using {device} device')
model.to(device)

train_losses = []
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if (i+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] - Step [{i+1}/{len(train_loader)}] - Loss: {running_loss/10:.4f}")
            running_loss = 0.0

    scheduler.step(running_loss)

# Evaluating the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to('cuda')
        labels = labels.to('cuda')

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on validation set: {(correct/total)*100:.2f}%")

torch.save(model.state_dict(), 'models/classifier.pt')

# model.eval()
# correct = 0
# total = 0
# with torch.no_grad():
#     for inputs, labels in test_loader:
#         inputs = inputs.to('cuda')
#         labels = labels.to('cuda')

#         outputs = model(inputs)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print(f"Accuracy on test set: {(correct/total)*100:.2f}%")
