import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image


# Load the CSV file
train_file_path = 'data/train_data.csv'  # Replace with your file path
test_file_path = 'data/class_test_from_vae_hard.csv'
train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

# Splitting the dataset into training and testing sets (80% - 20%)
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

# Display the sizes of the training and testing sets
print(f"Training set size: {len(train_data)}")
print(f"Testing set size: {len(test_data)}")

# Save the split datasets to new CSV files if needed
train_data.to_csv('train_data.csv', index=False)
val_data.to_csv('val_data.csv', index=False)


# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name)
        label = 0 if self.data.iloc[idx, 2] == 'M' else 1  # Assuming 'M' represents male and 'F' represents female
  # Convert label to a tensor

        if self.transform:
            image = self.transform(image)

        return image, label

# Define transformations for the images

class ToRGBTensor(object):
    def __call__(self, img):
        if img.mode == 'RGBA':
            r, g, b, a = img.split()
            img = Image.merge("RGB", (r, g, b))
        return transforms.ToTensor()(img)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ToRGBTensor(),
])

# Create instances of the custom dataset
train_dataset = CustomDataset(csv_file='data/train_data.csv', root_dir='/home/jacob/Documents/data/archive/SOCOFing/Real', transform=transform)
val_dataset = CustomDataset(csv_file='data/test_data.csv', root_dir= '/home/jacob/Documents/data/archive/SOCOFing/Real', transform=transform)
test_dataset = CustomDataset(csv_file= 'data/class_test_from_vae_hard.csv', root_dir= '/home/jacob/Documents/data/archive/SOCOFing/Repaired/Hard', transform=transform)
# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
# Define a simple CNN model
class SimpleCNN(torch.nn.Module):


  
   def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = torch.nn.Dropout(0.5)
        self.fc1 = torch.nn.Linear(128 * 16 * 16, 1024)
        self.fc2 = torch.nn.Linear(1024, 2)  # Assuming 601 output classes
        self.batch_norm1 = torch.nn.BatchNorm2d(32)
        self.batch_norm2 = torch.nn.BatchNorm2d(64)
        self.batch_norm3 = torch.nn.BatchNorm2d(128)

   def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.batch_norm2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.batch_norm3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(-1, 128 * 16 * 16)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Initialize the model, loss function, and optimizer
model = SimpleCNN()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, verbose=True)

model.to('cuda')

# Training the model
train_losses = []
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to('cuda')
        labels = labels.to('cuda')

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

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on test set: {(correct/total)*100:.2f}%")

