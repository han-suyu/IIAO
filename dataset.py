from tkinter import Y
from xmlrpc.client import NOT_WELLFORMED_ERROR
from torch.utils.data import Dataset
from PIL import Image
import os
import glob
import h5py
import cv2
import numpy as np
import scipy.io as io
from torchvision import transforms
from torch.utils.data import DataLoader


class IIAODataset(Dataset):
    def __init__(self, dir_path, scale):
        self.scale = scale
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])]
        )

        self.img_paths = []
        for img_path in glob.glob(os.path.join(dir_path, '*.jpg')):
            self.img_paths.append(img_path)
        # print(self.img_paths)

    def __getitem__(self, index):
        
        img_path = self.img_paths[index]
        img = Image.open(img_path).convert('RGB')


        
        density_path = img_path.replace('.jpg', '.h5').replace('images', 'density')
        with h5py.File(density_path, 'r') as hf:
            density = np.asarray(hf['density'])


        height, width = img.size[1], img.size[0]
        if height>5000 or width>5000:    # There are a few oversized images in the dataset, keeping the same operation as during training
                height = round(height * 0.8) 
                width = round(width * 0.8) 
                img = img.resize((width, height), Image.BILINEAR)
                density = cv2.resize(density,(width, height),interpolation=cv2.INTER_CUBIC) /0.8 /0.8
        now_hei, now_wid = height, width
        height = round(height / 16) * 16
        width = round(width / 16) * 16
        img = img.resize((width, height), Image.BILINEAR)
        img = self.transforms(img)
        density = cv2.resize(density,(width, height),interpolation=cv2.INTER_CUBIC) / (height/now_hei) /(width/now_wid)


        gt = np.sum(density)
     
    

        return img, gt, img_path

    
    def __len__(self):
        return len(self.img_paths)



if __name__=='__main__':
    IMG_DIR = 'datasets/jhu_crowd_v2.0/Overall/test/images' 
  
    val_dataset = IIAODataset(IMG_DIR,  scale = 8)
    print(len(val_dataset))
    val_loader = DataLoader(val_dataset, batch_size = 1, num_workers=0)
    print(len(val_loader))
  
