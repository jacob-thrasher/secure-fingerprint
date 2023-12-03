import os
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from network import VAE
from data import socofing_train_test_split_gen, SOCOFing_Gen, SOCOFing_Gen_Old
from train_test import train_vae

torch.manual_seed(69)

data_root = '/home/jacob/Documents/data/archive/SOCOFing/Altered'
gt_root = '/home/jacob/Documents/data/archive/SOCOFing/Real'

# train_samples, test_samples = socofing_train_test_split_gen(os.path.join(data_root, 'Altered-Hard'), test_size=0.2)
# train_dataset = SOCOFing_Gen_Old(os.path.join(data_root, 'Altered-Hard'), gt_root, train_samples)
# test_dataset = SOCOFing_Gen_Old(os.path.join(data_root, 'Altered-Hard'), gt_root, test_samples)

train_dataset = SOCOFing_Gen(data_root, gt_root, 'data/vae_train.csv', drop_difficulty=['Easy', 'Medium'], sample=1)
test_dataset = SOCOFing_Gen(data_root, gt_root, 'data/vae_test.csv', drop_difficulty=['Easy', 'Medium'], sample=1)

print(f'Loading data with:\n--> Train: {len(train_dataset)} samples\n--> Test : {len(test_dataset)} samples')

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

model = VAE(num_hiddens=128,
            num_residual_hiddens=32,
            num_residual_layers=2,
            z_dim=64)

if torch.cuda.is_available():
    print("\nUsing cuda device")
    device='cuda'
else:
    print('Using cpu')
    device = 'cpu'

model.to(device)

num_params = sum(p.numel() for p in model.parameters())
print(f'Training model with {num_params} parameters')

optim = Adam(model.parameters(), lr=1e-3, amsgrad=False)

train_vae(model, 
          train_dataloader, 
          test_dataloader, 
          optim, 
          device, 
          epochs=100, 
          beta=0,
          figpath='figures/csv_hard',
          modelpath='models/csv_hard')