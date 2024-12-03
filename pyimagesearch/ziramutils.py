# import the necessary packages
from sklearn.model_selection import train_test_split
from torch.utils import data
import torch
import pandas as pd
from PIL import Image
import os
import torchvision
import torchmetrics
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import glob
import mlflow
import rasterio


#TODO: Put those functions inside the Ziram dataset class
def get_dataloader(dataset, batch_size, shuffle=True):
	# create a dataloader
    dl = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)
    # return the data loader
    return dl


def train_val_split(dataset, batch_size,  validation_ratio=0.2,shuffle=True):
    # grab the total size of the dataset
    totalSize = len(dataset)
    # perform training and validation split
    (trainIdx, valIdx) = train_test_split(list(range(totalSize)),
        test_size=validation_ratio)
    trainDataset = data.Subset(dataset, trainIdx)
    valDataset = data.Subset(dataset, valIdx)
    # return training and validation dataset
    return (get_dataloader(trainDataset, batch_size=batch_size, shuffle=shuffle),
             get_dataloader(valDataset, batch_size=batch_size, shuffle=shuffle))


# Define a generic dataset class for Ziram data used to present data to the model
class ZiramDataset(data.Dataset):
    """Default Dataset class adapted for the Ziram dataset."""
    # We want the dataset to contain the filepath, set attribution, label, 
    def __init__(self, dataset_path, mode,           # Path of the datset containing the image repartitions and labels
                 im_size, mean, std,            # Those values are pre-defined in the pretrained model usecases
                 label='label', filename='filename',    # Details of the columns to use for the dataset
                 augment=False, exlude_controls=False):  # split in ['training', 'test']
        assert (mode in ['training', 'test']), 'Wrong value for dataset. Choose a dataset between \'training\' and \'test\''

        # files = glob.glob(os.path.join(path, dataset, '*png'))
        # files_dict= {os.path.basename(file).split('.')[0] : file for file in files}
        try:
            dataset_info = pd.read_csv(dataset_path)
            dataset_info = dataset_info[dataset_info['set'] == mode ]
            # dataset_info = dataset_info[dataset_info[filename].isin(files_dict.keys())].dropna(subset=[label])

        except KeyError:
            if filename not in dataset_info.columns:
                raise KeyError(f'Column `{filename}` not in dataset file !')
            elif label not in dataset_info.columns:
                raise KeyError(f'Column `{label}` not in dataset file !')
            elif 'set' not in dataset_info.columns:
                raise KeyError(f'Column `set` not in dataset file !')
            else:
                raise KeyError
        files = [file for file in dataset_info[filename].unique().tolist() if os.path.exists(file)]
        assert len(files) > 1, 'Could not find the files'
        if len(files) != dataset_info[filename].nunique():
            print(f'[WARN] Number of files ({len(files)}) and dataset files ({dataset_info[filename].nunique()}) not corresponding (files or label missing)')

        # dataset_info['img_path'] = dataset_info[filename].map(files_dict)
        dataset_info = dataset_info[dataset_info[filename].isin(files)]
        dataset_info.dropna(subset=[filename, label], inplace=True)

        try:
            dataset_info['label_categorical'] = dataset_info[label].astype(int)
        except:
            dict_labels = {val:n for (n, val) in enumerate(dataset_info[label].sort_values().unique())}
            dataset_info['label_categorical'] = dataset_info[label].map(dict_labels).astype(int)
            print(f'[INFO] The labels were cast as integer according to the following mapping {dict_labels}')

        dataset_info['label_regression'] =(dataset_info['label_categorical'] / dataset_info['label_categorical'].max()).astype(float)
        dataset_info['img_path'] = dataset_info[filename]
        # Set the attributes
        self.dataset = dataset_info
        # self.folder = path
        self.im_size = (im_size, im_size)
        print(dataset_info['label_regression'].unique())
        self.__data = dataset_info[['img_path', 'label_categorical' , 'label_regression']].to_dict('records')
        self.mean, self.std = mean, std
        self.filename = filename
        self.num_label = dataset_info['label_categorical'].nunique()
        # self.valSplit = validation_ratio
        # self.batch_size = batch_size
        # self.shuffle = shuffle
        self.augment = augment      # Wether to randomly transform images or not

    
    def __getitem__(self, index):
        """Default iterator function."""
        element = self.__data[index]
        image_path = str(element['img_path'])
        
        # Reading and image porcessing (opencv version)
        # cv_image = cv2.imread(image_path)
        # cv_image = cv2.resize(cv_image, self.im_size).astype(np.uint8)
        # cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        # out = torchvision.transforms.functional.to_tensor(cv_image)
        # pil_image = Image.open(image_path)


        #Instead of using torch to open the image we use rasterio to be able to open tif too
        # img = torchvision.io.read_image(image_path)
        img = rasterio.open(image_path)
        img = torchvision.transforms.ToTensor()(img.read().astype(np.uint8))
        img = torchvision.transforms.ConvertImageDtype(torch.float32)(img)
        img = img.permute(1,0,2)
        img = torchvision.transforms.Resize(self.im_size, antialias=True)(img)
        img = torch.cat([img, img, img], dim=0)
        img = torchvision.transforms.Normalize(self.mean, self.std)(img)


        # img = rasterio.open(image_path)
        # img = torchvision.transforms.ToTensor()(img.read().astype(np.uint8))
        # img = torchvision.transforms.ConvertImageDtype(torch.float32)(img)

        # img = torchvision.transforms.Resize((256,256), antialias=True)(img)
        # img = torch.cat([img, img, img], dim=0)
        # img = torchvision.transforms.Normalize(mean, std)(img)

        
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
        

        element['image'] = img
        # return self.train_val_split(element, valSplit=self.valSplit)

        return element

    def __len__(self):
        """Default length internal function."""
        return len(self.__data)

# ##If problems drop from here
#     def train_val_split(self, dataset, valSplit=0.2):
#         # grab the total size of the dataset
#         totalSize = len(dataset)
#         # perform training and validation split
#         (trainIdx, valIdx) = train_test_split(list(range(totalSize)),
#             test_size=valSplit)
#         trainDataset = data.Subset(dataset, trainIdx)
#         valDataset = data.Subset(dataset, valIdx)
#         # return training and validation dataset
#         trainLoader = self.get_dataloader(trainDataset, batchSize=self.batch_size, shuffle=self.shuffle)
#         valLoader = self.get_dataloader(valDataset, batchSize=self.batch_size, shuffle=self.shuffle)
#         return (trainLoader, valLoader)



#     def get_dataloader(self, dataset, batchSize, shuffle=True):
#         # create a dataloader
#         dl = data.DataLoader(dataset, batch_size=batchSize, shuffle=shuffle, drop_last=False)
#         # return the data loader
#         return dl