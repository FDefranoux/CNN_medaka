# import the necessary packages
from sklearn.model_selection import train_test_split
from torch.utils import data
import torch
import pandas as pd
from PIL import Image
import os
import torchvision
import torchmetrics
from torchvision.transforms import Normalize
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import glob
import mlflow


def get_dataloader(dataset, batchSize, shuffle=True):
	# create a dataloader
    dl = data.DataLoader(dataset, batch_size=batchSize, shuffle=shuffle, drop_last=False)
    # return the data loader
    return dl


def train_val_split(dataset, valSplit=0.2):
    # grab the total size of the dataset
    totalSize = len(dataset)
    # perform training and validation split
    (trainIdx, valIdx) = train_test_split(list(range(totalSize)),
        test_size=valSplit)
    trainDataset = data.Subset(dataset, trainIdx)
    valDataset = data.Subset(dataset, valIdx)
    # return training and validation dataset
    return (trainDataset, valDataset)


# Define a generic dataset class for Ziram data used to present data to the model
class ZiramDataset(data.Dataset):
    """Default Dataset class adapted for the Ziram dataset."""
    
    def __init__(self, path: str, dataset: str, im_size: int, mean: tuple, std: tuple, filename: str, label: str, num_class: int,
                 augment: bool = False, exlude_controls: bool = False):  # split in ['training', 'test']
        assert (dataset in ['training_set', 'test_set']), 'Wrong value for dataset. Choose a dataset between \'training_set\' and \'test_set\''
        dataset_info = pd.read_csv(os.path.join(path, 'Repartition_images_sets.csv'))
        files = glob.glob(os.path.join(path, dataset, '*png'))
        files_dict= { os.path.basename(file).split('.')[0] : file for file in files}
        dataset_info = dataset_info[dataset_info['folder'] == dataset]
        try:
            dataset_info = dataset_info[dataset_info[filename].isin(files_dict.keys())].dropna(subset=[label])
        except KeyError:
            if filename not in dataset_info.columns:
                raise KeyError(f'Column {filename} not in base info file !')
            elif label not in dataset_info.columns:
                raise KeyError(f'Column {label} not in base info file !')
            else:
                raise KeyError

        if len(files) != dataset_info[filename].nunique():
            print(f'[WARN] Number of files ({len(files)}) and dataset files ({dataset_info[filename].nunique()}) not corresponding (files or label missing)')

        dataset_info['img_path'] = dataset_info[filename].map(files_dict)
        try:
            dataset_info['label_class'] = dataset_info[label].astype(int)
        except:
            dict_labels = {val:n for (n, val) in enumerate(dataset_info[label].sort_values().unique())}
            dataset_info['label_class'] = dataset_info[label].map(dict_labels).astype(int)
            print(f'[INFO] The labels were cast as integer according to the following mapping {dict_labels}')

        dataset_info['label_reg'] =(dataset_info['label_class'] / dataset_info['label_class'].max()).astype(float)
        num_label_dataset = dataset_info['label_class'].nunique()
        assert int(num_label_dataset) == int(num_class), f'Numbers of labels ({num_label_dataset}) not corresponding to class number ({num_class})'
        
        # Set the attributes
        self.dataset = dataset
        self.folder = path
        self.im_size = (im_size, im_size)
        self.__data = dataset_info[['img_path', 'label_class' , 'label_reg']].to_dict('records')
        self.mean, self.std = mean, std
        self.augment = augment      # Wether to randomly transform images or not
        self.filename = filename

    
    def __getitem__(self, index):
        """Default iterator function."""
        element = self.__data[index]
        image_path = os.path.join(self.folder, self.dataset, os.path.basename(str(element['img_path'])))
        
        # Reading and image porcessing (opencv version)
        # cv_image = cv2.imread(image_path)
        # cv_image = cv2.resize(cv_image, self.im_size).astype(np.uint8)
        # cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        # out = torchvision.transforms.functional.to_tensor(cv_image)

        # pil_image = Image.open(image_path)
        img = torchvision.io.read_image(image_path)
        img = torchvision.transforms.ConvertImageDtype(torch.float32)(img)
        img = torchvision.transforms.Resize(self.im_size, antialias=True)(img)
        img = torch.cat([img, img, img], dim=0)
        img = torchvision.transforms.Normalize(self.mean, self.std)(img)

        # pil_image = pil_image.convert('L')
        # print('description1_bis', np.asarray(pil_image).dtype, np.min(np.asarray(pil_image)), np.max(np.asarray(pil_image)))
        # pil_image = pil_image.resize(self.im_size)  #.astype(np.uint8)
        # pil_image = pil_image.convert('RGB')
        # print(pil_image.size)
        # print(np.asarray(pil_image).max())
        # print(np.asarray(pil_image).min())
        # pil_image = pil_image.resize(self.im_size)  #.astype(np.uint8)
        # print(pil_image.size)
        # print(np.asarray(pil_image).max())
        # print(np.asarray(pil_image).min())

        # print('description2', np.asarray(pil_image).dtype,  np.min(np.asarray(pil_image)), np.max(np.asarray(pil_image)))
        # image_array = np.asarray(pil_image).astype(np.float32)
        # print('description3', np.asarray(image_array).dtype,  np.min(np.asarray(image_array)), np.max(np.asarray(image_array)))
        # pil_image = pil_image.convert(mode='L')
        # out = torch.from_numpy(np.array(pil_image).astype(np.float32))

        # out = torchvision.transforms.functional.to_tensor(image_array)
        if self.augment:
            img = torchvision.transforms.RandomRotation(90)(img)
            
        # print('[VERIF] array description\n', stats.describe(image_array, axis=0))

        # plt_image = plt.imshow((img.permute(1, 2, 0)/torch.max(torch.abs(img))+1)/2, cmap='Greys')
        # plt.imsave(f'image_index{index}.png', plt_image)
        # print('[VERIF] image saved')
        # sys.exit(0)

        
        # out shape is [3, 256, 256]
        # plt.show()

        element['image'] = img

        return element

    def __len__(self):
        """Default length internal function."""
        return len(self.__data)


