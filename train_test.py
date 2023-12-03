import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure


#########
#  VAE  #
#########

def create_img_grid(model, dataloader, filename, device, epoch):
    model.eval()
    X, Y = next(iter(dataloader))
    X = X.to(device)
    Y = Y.to(device)
    X_recon, _, _ = model(X)

    n = min(X.size(0), 8)
    comparison = torch.cat([Y[:n]*255, X[:n]*255, X_recon[:n]*255])

    grid = make_grid(comparison)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    
    plt.figure()
    plt.imshow(ndarr)
    plt.title(f'Sample images -- Epoch {epoch}')
    plt.axis('off')
    plt.savefig(filename)
    plt.close()

def vae_loss(recon_x, x, mu, logvar, beta=0):   
    # recon_loss = F.binary_cross_entropy(torch.sigmoid(recon_x), torch.sigmoid(x), reduction='sum')
    recon_loss = F.mse_loss(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + beta * KLD

def train_step_vae(model, dataloader, optim, device, beta):
    model.train()
    tot_loss = 0
    for _, (X, Y) in enumerate(dataloader):
        X = X.to(device)
        Y = Y.to(device)
        
        optim.zero_grad()
        X_recon, mu, logvar = model(X)
        loss = vae_loss(X_recon, Y, mu, logvar, beta)
        tot_loss += loss.item()

        loss.backward()
        optim.step()

    return tot_loss / len(dataloader)

def test_step_vae(model, dataloader, device, beta):
    model.eval()
    mse_tot = 0
    psnr_tot = 0
    ssim_tot = 0
    for _, (X, Y) in enumerate(dataloader):
        X = X.to(device)
        Y = Y.to(device)
        
        X_recon, mu, logvar = model(X)

        mse = vae_loss(X_recon, Y, mu, logvar, beta)
        psnr = peak_signal_noise_ratio(X_recon, Y)
        ssim = structural_similarity_index_measure(X_recon, Y)

        mse_tot += mse.item()
        psnr_tot += psnr.item()
        ssim_tot += ssim.item()

    return mse_tot / len(dataloader), psnr_tot / len(dataloader), ssim_tot / len(dataloader)

def train_vae(model, 
              train_dataloader, 
              test_dataloader, 
              optim, 
              device='cuda', 
              epochs=100, 
              beta=0,
              figpath='figures',
              modelpath='models'):
    all_train_loss = []
    all_test_loss = []
    all_psnr = []
    all_ssim = []
    best_test_loss = 100000000
    for epoch in range(epochs+1):
        train_loss = train_step_vae(model, train_dataloader, optim, device, beta)
        test_loss, psnr, ssim = test_step_vae(model, test_dataloader, device, beta)

        all_train_loss.append(train_loss)
        all_test_loss.append(test_loss)
        all_psnr.append(psnr)
        all_ssim.append(ssim)


        if epoch % 10 == 0:
            create_img_grid(model, test_dataloader, filename=os.path.join(figpath, f'output_e{epoch}.png'), device=device, epoch=epoch)
            torch.save(model.state_dict(), os.path.join(modelpath, f'VAE_e{epoch}.pt'))

        print(f'Epoch [{epoch+1}]/[{epochs}]:')
        print(f'--> Train loss: {train_loss}')
        print(f'--> Test loss : {test_loss}')
        print(f'-->      psnr : {psnr}')
        print(f'-->      ssim : {ssim}')

        plt.figure()
        plt.plot(all_train_loss, color='blue', label='Train loss')
        plt.plot(all_test_loss, color='red', label='Test loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('MSE loss')
        plt.legend()
        plt.savefig(os.path.join(figpath, 'loss.png'))
        plt.close()

        plt.figure()
        plt.plot(all_psnr, color='green')
        plt.xlabel('Epoch')
        plt.ylabel('PSNR')
        plt.title('Peak Signal to Noise Ratio')
        plt.savefig(os.path.join(figpath, 'psnr.png'))
        plt.close()

        plt.figure()
        plt.plot(all_ssim, color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('SSIM')
        plt.title('Structural Similarity Index Measure')
        plt.savefig(os.path.join(figpath, 'ssim.png'))
        plt.close()
