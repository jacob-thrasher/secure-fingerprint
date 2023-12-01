import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


#########
#  VAE  #
#########

def create_img_grid(model, dataloader, filename, device):
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
    plt.axis('off')
    plt.savefig(filename)

def vae_loss(recon_x, x, mu, logvar, beta=0):   
    # recon_loss = F.binary_cross_entropy(recon_x.view(-1, 10000), x.view(-1, 10000), reduction='sum')
    recon_loss = F.mse_loss(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + beta * KLD

def train_step_vae(model, dataloader, optim, device):
    model.train()
    tot_loss = 0
    for _, (X, Y) in enumerate(dataloader):
        X = X.to(device)
        Y = Y.to(device)
        
        optim.zero_grad()
        X_recon, mu, logvar = model(X)
        loss = vae_loss(X_recon, Y, mu, logvar)
        tot_loss += loss.item()

        loss.backward()
        optim.step()

    return tot_loss / len(dataloader)

def test_step_vae(model, dataloader, device):
    model.eval()
    tot_loss = 0
    for _, (X, Y) in enumerate(dataloader):
        X = X.to(device)
        Y = Y.to(device)
        
        X_recon, mu, logvar = model(X)
        loss = vae_loss(X_recon, Y, mu, logvar)
        tot_loss += loss.item()

    return tot_loss / len(dataloader)

def train_vae(model, train_dataloader, test_dataloader, optim, device='cuda', epochs=100):
    all_train_loss = []
    all_test_loss = []
    for epoch in range(epochs):
        train_loss = train_step_vae(model, train_dataloader, optim, device)
        test_loss = test_step_vae(model, test_dataloader, device)

        all_train_loss.append(train_loss)
        all_test_loss.append(test_loss)

        if epoch % 10 == 0:
            create_img_grid(model, test_dataloader, filename=f'figures/output_e{epoch}.png', device=device)

        print(f'Epoch [{epoch+1}]/[{epochs}]:')
        print(f'--> Train loss: {train_loss}')
        print(f'--> Test loss: {test_loss}')

        plt.figure()
        plt.plot(all_train_loss, color='blue', label='Train loss')
        plt.plot(all_test_loss, color='red', label='Test loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('VAE loss')
        plt.legend()
        plt.savefig('figures/loss.png')
