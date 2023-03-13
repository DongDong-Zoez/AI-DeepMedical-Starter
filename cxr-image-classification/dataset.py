import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils.data import Dataset
from PIL import Image

# Third-party packages

def preprocessor(base_root, csv_name):
    df = pd.read_csv(os.path.join(base_root, csv_name))
    category = ['NORMAL','PNEUMONIA']
    for pathology in category :
        df[pathology] = df.label.apply(lambda x: 1 if pathology in x else 0)
    return df

class CXRDataset(Dataset):

    def __init__(self, df, transforms, normalize=True, mode="train"):
        
        self.img_files = df.path.to_list()
        self.df = df
        self.transforms = transforms
        self.mode = mode
        self.normalize = normalize
    
    def __getitem__(self, x):
        row = self.df.iloc[x, :].values
        *_, normal, pneumonia = row
        image = Image.open(self.img_files[x]).convert('RGB')
        image = np.array(image)
        if self.normalize:
            image = image - image.min()
            image = image / image.max()
        if self.transforms is not None:
            image = self.transforms(image=image)['image']
        # Convert the pixel value to [0, 1]
        if self.mode == "train":
            one_hot_label = [normal, pneumonia]
            return image, torch.tensor(one_hot_label)
        else:
            return image
    
    def __len__(self):
        return len(self.img_files)