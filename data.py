import os
import random
import torchvision.transforms.functional as F
import pandas as pd
import csv

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

def class_csv_from_vae_out(root, dst, csvname):
    '''
    Args:
        root (pathLike): path to VAE outputs
        dst (pathLike): path to csv output
        csvname (str): output name
    '''

    f = open(os.path.join(dst, csvname), 'w', newline='')
    writer = csv.writer(f)
    writer.writerow(["Image Name", "Number", "Gender", "Hand", "Finger", "Alteration"])

    for filename in tqdm(os.listdir(root)):
        attr = filename.split('_')

        writer.writerow([filename, attr[0], attr[2], attr[3], attr[4], attr[6].split('.')[0]])

    f.close()


# TODO: Add support for altered fingerprints
# TODO: Add validation support
def get_train_test_split_socofing(root, test_size=0.2, valid_size=0.2, split_method='standard'):
    '''
    Divides dataset into train, test, and validation splits

    Args:
        root: (pathLike) path to data root
        test_size: (float) % of subjects to allocate to test set
        valid_size: (float) % of train set to allocate to validation

    Returns:
        Tuple: (train_dataset, test_dataset, valid_dataset)
    '''

    assert split_method in ['standard', 'subject'], f'parameter split_method should be in [standard, subject], got {split_method}'

    files = os.listdir(root)
    subject_ids = list(range(1, 601))
    if split_method == 'subject':
        # Divide train/test by subject ID to avoid data leakage
        num_test_samples = int(len(subject_ids) * test_size)
        test_subjects = random.sample(subject_ids, num_test_samples)
        train_subjects = [x for x in subject_ids if x not in test_subjects]
            
        return train_subjects, test_subjects
    elif split_method == 'standard':
        num_test_samples = int(len(files) * test_size)

def socofing_train_test_split_gen(root, test_size):
    files = os.listdir(root)

    n_test_samples = test_size
    if test_size < 1:
        n_test_samples = int(len(files) * test_size)

    test_samples = random.sample(files, n_test_samples)
    train_samples = [x for x in files if x not in test_samples]
    return train_samples, test_samples


######################
# CLASSIFIER CLASSES #
######################

class ToRGBTensor(object):
    def __call__(self, img):
        if img.mode == 'RGBA':
            r, g, b, a = img.split()
            img = Image.merge("RGB", (r, g, b))
        return transforms.ToTensor()(img)

class SOCOFing_class(Dataset):
    def __init__(self, csv_file, root_dir, return_filename=False):
        self.return_filename = return_filename
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ToRGBTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        label = 0 if self.data.iloc[idx, 2] == 'M' else 1  # Assuming 'M' represents male and 'F' represents female
        # Convert label to a tensor


        image = self.transform(image)

        if self.return_filename: return image, label, self.data.iloc[idx, 0]
        return image, label

###############
# VAE CLASSES #
###############
    
class SOCOFing_Gen(Dataset):
    '''
        Load SOCOFing images and ground truth data

        Args:
            dataroot: (pathLike) path to training images
            gtroot: (pathLike) path to ground truth
        '''
    def __init__(self, data_root, gt_root, data_csv, drop_difficulty=[], resize=(100, 100), sample=1):
        self.df = pd.read_csv(data_csv)

        for d in drop_difficulty:
            assert d in ['Easy', 'Medium', 'Hard'], f'drop_difficulty expects inputs to be in [Easy, Medium, Hard], got {d}'
            self.df = self.df[self.df['Difficulty'] != d]

        self.df = self.df.sample(frac=sample, random_state=1)
        
        # TEMP
        # self.df = self.df[self.df['isCorrect'] == False]

        self.data_root = data_root
        self.gt_root = gt_root        

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resize),
            transforms.Normalize((0.5), (1.0))
        ])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        train_name = row.Image_Name
        difficulty = row.Difficulty

        attr = train_name.split('_')
        gt_name = attr[0] + '__' + attr[2] + '_' + attr[3] + '_' + attr[4] + '_finger.BMP'

        train_img_path = os.path.join(self.data_root, f'Altered-{difficulty}', train_name)
        gt_path = os.path.join(self.gt_root, gt_name)
        train_img = self.transform(Image.open(train_img_path).convert('L')).squeeze()
        gt_img = self.transform(Image.open(gt_path).convert('L')).squeeze()

        return train_img.unsqueeze(0), gt_img.unsqueeze(0)
    

class SOCOFing_Gen_Old(Dataset):
    '''
        Load SOCOFing images and ground truth data

        Args:
            dataroot: (pathLike) path to training images
            gtroot: (pathLike) path to ground truth
        '''
    def __init__(self, data_root, gt_root, data_samples, resize=(100, 100)):
        self.data_root = data_root
        self.data = data_samples
        self.gt_root = gt_root        

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resize),
            transforms.Normalize((0.5), (1.0))
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        train_name = self.data[idx]
        attr = train_name.split('_')
        gt_name = attr[0] + '__' + attr[2] + '_' + attr[3] + '_' + attr[4] + '_finger.BMP'

        train_img_path = os.path.join(self.data_root, train_name)
        gt_path = os.path.join(self.gt_root, gt_name)
        train_img = self.transform(Image.open(train_img_path).convert('L')).squeeze()
        gt_img = self.transform(Image.open(gt_path).convert('L')).squeeze()

        return train_img.unsqueeze(0), gt_img.unsqueeze(0)
    
