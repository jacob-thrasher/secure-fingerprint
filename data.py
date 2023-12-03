import os
import random
import torchvision.transforms.functional as F
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

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

class SOCOFing_Class(Dataset):
    def __init__(self, root, subjects, resize=(100, 100)):
        '''

        Load SOCOFing dataset and labels based on the following naming convention

        Example: 1__M_Left_index_finger_Obl.BMP
            "1"     --> Subject ID
            "M"     --> Sex (Male/Female)
            "Left"  --> Hand (Left/Right)
            "index" --> Finger (little, ring, middle, index, thumb)
            "Obl"   --> Alteration type - Note: not present in "real" data
                            (Obl - obliteration, CR - central rotation, Zcut)
            From: https://arxiv.org/ftp/arxiv/papers/1807/1807.10609.pdf

        Args:
            root: (pathLike) path to dataset
            subjects: (list[str]) list of subject IDs
        '''

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resize),
            transforms.Normalize((0.5), (1.0))
        ])

        files = os.listdir(root)
        samples = [f for f in files if int(f.split('_')[0]) in subjects]
        self.data = []

        for s in tqdm(samples, 'Loading data'):
            img_path = os.path.join(root, s)
            # Load image
            img = transform(Image.open(img_path).convert('L')).squeeze()

            # Get attributes
            filename = s.split('.')[0]
            attributes = filename.split('_')
            subject_labels = {
                "id": attributes[0],
                "sex": attributes[2],
                "hand": attributes[3],
                "finger": attributes[4],
                "alteration": attributes[6] if len(attributes) == 7 else None,
                "path": img_path,
                "img": img
            }
            self.data.append(subject_labels)

    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]['img'].unsqueeze(0)
    
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
    
