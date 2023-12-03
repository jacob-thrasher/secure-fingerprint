import os
import torch
import pandas as pd
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from network import VAE
from tqdm import tqdm


data_root = '/home/jacob/Documents/data/archive/SOCOFing/Altered/Altered-Hard'
dst = '/home/jacob/Documents/data/archive/SOCOFing/Repaired/Hard'

df = pd.read_csv('data/vae_test.csv')
df = df[df['Difficulty'] == 'Hard']

subjects = list(df.Image_Name)

transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((100, 100)),
            transforms.Normalize((0.5), (1.0))
        ])

model = VAE(num_hiddens=128,
            num_residual_hiddens=32,
            num_residual_layers=2,
            z_dim=64)
model.load_state_dict(torch.load('models/csv_hard/VAE_e100.pt'))
model.to('cuda')
model.eval()

for filename in tqdm(subjects):
    path = os.path.join(data_root, filename)
    img = transform(Image.open(path).convert('L'))
    img = img.to('cuda')


    out = model(img.unsqueeze(0))

    save_image(out[0].squeeze().detach().cpu()*255, os.path.join(dst, filename))