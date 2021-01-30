import numpy as np
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


def preprocess(image_list):
    """ Normalize Image and Permute (N,H,W,C) to (N,C,H,W)
    Args:
      image_list: List of images (9000, 32, 32, 3)
    Returns:
      image_list: List of images (9000, 3, 32, 32)
    """
    image_list = np.array(image_list)
    image_list = np.transpose(image_list, (0, 3, 1, 2))
    image_list = (image_list / 255.0) * 2 - 1
    image_list = image_list.astype(np.float32)
    return image_list

class Image_Dataset(Dataset):
    def __init__(self, image_list,train=True):
        self.image_list = image_list
        self.transform = transforms.Compose([
        transforms.RandomAffine(10,translate=(0.1,0.1),scale=(0.9,1.1)),
        transforms.RandomHorizontalFlip()
        ])
        self.train = train
        return
    def __len__(self):
        return len(self.image_list)
    def __getitem__(self , idx):
        image = self.image_list[idx]
        image = Image.fromarray(np.uint8(image))
        if self.train == True:
            image = self.transform(image)
        image = np.asarray(image)
        image = np.transpose(image , (2 , 0 , 1))
        image = (image / 255)*2-1
        return image.astype(np.float32)

def count_parameters(model, only_trainable=False):
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
