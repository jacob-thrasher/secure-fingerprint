import torch.functional as F
import numpy as np

# def train_step(model, epochs, dataloader, optim, device):
#     for i, (X, Y) in enumerate(dataloader):
#         X = X.to(device)
#         Y = Y.to(device)
        
#         optim.zero_grad()
#         X_recon = model(X)
#         loss = F.mse_loss(X_recon, X)

#         loss.backward()
#         optim.step()