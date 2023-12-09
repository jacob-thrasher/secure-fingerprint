import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim import Adam
from network import VAE
from data import socofing_train_test_split_gen, SOCOFing_Gen, SOCOFing_Gen_Old
from train_test import train_vae


torch.manual_seed(69)

data_root = 'D:\\Big_Data\\SOCOFing\\Altered'
gt_root =  'D:\\Big_Data\\SOCOFing\\Real'
experiments = ['Easy', 'Hard']
figpath = 'figures\\50-50'
modelpath = 'models\\50-50'
# train_samples, test_samples = socofing_train_test_split_gen(os.path.join(data_root, 'Altered-Hard'), test_size=0.2)
# train_dataset = SOCOFing_Gen_Old(os.path.join(data_root, 'Altered-Hard'), gt_root, train_samples)
# test_dataset = SOCOFing_Gen_Old(os.path.join(data_root, 'Altered-Hard'), gt_root, test_samples)

psnr_all = []
ssim_all = []
for i in range(2):
    exp = experiments[i]
    if exp == 'Easy': drop = ['Medium', 'Hard']
    else: drop = ['Easy', 'Medium']

    print(f"\n\nStating Experiment: {exp}\n---------------------\n")

    train_dataset = SOCOFing_Gen(data_root, gt_root, 'data/vae_train_50.csv', drop_difficulty=drop, sample=1)
    test_dataset = SOCOFing_Gen(data_root, gt_root, 'data/vae_test_50.csv', drop_difficulty=drop, sample=1)

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

    psnr, ssim = train_vae(model, 
                        train_dataloader, 
                        test_dataloader, 
                        optim, 
                        device, 
                        epochs=100, 
                        beta=0,
                        figpath=os.path.join(figpath, exp),
                        modelpath=os.path.join(modelpath, exp))
    
    psnr_all.append(psnr)
    ssim_all.append(ssim)

plt.figure()
plt.plot(psnr_all[0], color='blue', label='Easy')
plt.plot(psnr_all[1], color='red', label='Hard')
plt.xlabel('Epoch')
plt.ylabel('PSNR')
plt.title('PSNR score Easy vs Hard')
plt.legend()
plt.savefig(os.path.join(figpath, 'psnr.png'))
plt.close()

plt.figure()
plt.plot(ssim_all[0], color='green', label='Easy')
plt.plot(ssim_all[1], color='orange', label='Hard')
plt.xlabel('Epoch')
plt.ylabel('SSIM')
plt.title('SSIM score Easy vs Hard')
plt.legend()
plt.savefig(os.path.join(figpath, 'ssim.png'))
plt.close()
