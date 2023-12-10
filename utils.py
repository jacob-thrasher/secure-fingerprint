import os
import torch
import pandas as pd
import ast
import csv
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from network import VAE, SimpleCNN
from tqdm import tqdm
from data import SOCOFing_class, SOCOFing_Gen
from torchmetrics.functional.image import structural_similarity_index_measure

def repair_fingerprints(data_root, dst, csvpath, modelpath, difficulty):
    '''
    Repairs altered fingerprints using pretrained VAE

    Args:
    - dataroot (pathLike)  : Path to altered files
    - dst (pathLike)       : Path to save VAE output
    - cvspath (pathLike)   : path to csv containing file information
    - modelpath (pathLike) : Path to pretrained VAE model
    - Difficulty (str)     : Altered difficulty
    '''

    assert difficulty in ['Easy', 'Medium', "Hard"], f'Expected parameter difficulty to be in [Easy, Medium, Hard], got {difficulty}'

    df = pd.read_csv(csvpath)
    df = df[df['Difficulty'] == difficulty]

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
    model.load_state_dict(torch.load(modelpath))
    model.to('cuda')
    model.eval()

    for filename in tqdm(subjects):
        path = os.path.join(data_root, filename)
        img = transform(Image.open(path).convert('L'))
        img = img.to('cuda')


        out = model(img.unsqueeze(0))

        save_image(out[0].squeeze().detach().cpu()*255, os.path.join(dst, filename))

def sort_predictions(data_root, dst, csvpath, modelpath, difficulty='Easy'):
    '''
    Classifies images and sorts based on correct and incorrect classifications

    Args:
    - dataroot (pathLike)  : Path to image files
    - dst (pathLike)       : Path to results
    - cvspath (pathLike)   : path to csv containing file information
    - modelpath (pathLike) : Path to pretrained classifier model
    - Difficulty (str)     : Altered difficulty
    '''

    dataset = SOCOFing_class(csvpath, data_root, return_filename=True)
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = SimpleCNN()
    model.load_state_dict(torch.load(modelpath))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    f = open(dst, 'w', newline='')
    writer = csv.writer(f)
    writer.writerow(["Image_Name", "isCorrect", "Difficulty"])

    for i, (X, y, filename) in enumerate(dataset):
        X = X.to(device).unsqueeze(0)
        out = model(X).squeeze()
        pred = torch.argmax(out, dim=0).item()

        writer.writerow([filename, pred == y, difficulty])

    f.close()

def ssim_evaluation(data_root, gt_root, csvpath, modelpath):

    dataset = SOCOFing_Gen(data_root, gt_root, csvpath)
    dataloader = DataLoader(dataset, batch_size=32)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = VAE(num_hiddens=128,
                num_residual_hiddens=32,
                num_residual_layers=2,
                z_dim=64)
    model.load_state_dict(torch.load(modelpath))
    model.to(device)
    model.eval()

    ssim = 0
    for i, (X, Y) in enumerate(dataloader):
        X = X.to(device)
        Y = Y.to(device)

        recon, _, _ = model(X)
        ssim += structural_similarity_index_measure(recon, Y).item()

    print(f'SSIM:', ssim / len(dataloader))


ssim_evaluation(data_root='D:\\Big_Data\\SOCOFing\\Altered',
                gt_root='D:\\Big_Data\\SOCOFing\\Real',
                csvpath='files/Pred_Easy.csv',
                modelpath='models/80-20/Easy/VAE_e100.pt')

# sort_predictions(data_root='D:\\Big_Data\\SOCOFing\\Repaired\\Repaired-Easy_20',
#                 dst='files/Pred_Easy.csv',
#                 csvpath='data/class_from_vae/Easy_20.csv',
#                 modelpath='models/classifier.pt',
#                 difficulty='Easy')

# sort_predictions(data_root='D:\\Big_Data\\SOCOFing\\Repaired\\Repaired-Hard_20',
#                 dst='files/Pred_Hard.csv',
#                 csvpath='data/class_from_vae/Hard_20.csv',
#                 modelpath='models/classifier.pt',
#                 difficulty='Hard')

