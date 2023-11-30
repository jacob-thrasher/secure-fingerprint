import os
import random
import torchvision.transforms.functional as F

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

# TODO: Add support for altered fingerprints
# TODO: Add validation support
def get_train_test_split_socofing(root, test_size=0.2, valid_size=0.2):
    '''
    Divides dataset into train, test, and validation splits

    Args:
        root: (pathLike) path to data root
        test_size: (float) % of subjects to allocate to test set
        valid_size: (float) % of train set to allocate to validation

    Returns:
        Tuple: (train_dataset, test_dataset, valid_dataset)
    '''

    files = os.listdir(root)
    subject_ids = list(range(1, 601))

    # Divide train/test by subject ID to avoid data leakage
    num_test_samples = int(len(subject_ids) * test_size)
    test_subjects = random.sample(subject_ids, num_test_samples)
    train_subjects = [x for x in subject_ids if x not in test_subjects]
        
    return train_subjects, test_subjects

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
    def __init__(self, data_root, gt_root, resize=(100, 100)):
        self.data_root = data_root
        self.data = os.listdir(data_root)
        self.gt_root = gt_root
        self.gt = os.listdir(gt_root)

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

        return train_img, gt_img
    
