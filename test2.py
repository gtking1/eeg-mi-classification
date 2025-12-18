import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import pandas as pd
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations, WindowOperations
from sklearn.metrics import f1_score
import torch.optim

class ToTensor(object):
    def __call__(self, sample):
        window, labels = sample['window'], sample['labels']
        return {'window': torch.tensor(window.values).to(torch.float32), 
                'labels': torch.tensor(labels).to(torch.float32)}#.values)}

class AbsoluteValue(object):
    def __call__(self, sample):
        window, labels = sample['window'], sample['labels']
        return {'window': torch.abs(window), 
                'labels': labels}

class MinNormalize(object):
    def __call__(self, sample):
        window, labels = sample['window'], sample['labels']
        for channel in range(len(window)):
            #print(len(window))
            min = torch.min(window[channel])
            window[channel] = window[channel] - min
            window[channel] = window[channel] / torch.max(window[channel])
            window[channel] = torch.nan_to_num(window[channel])
        return {'window': (window.unsqueeze(0)), 
                'labels': labels}

# class FixLabels(object):
#     def __call__(self, sample):
#         window, labels = sample['window'], sample['labels']
#         if labels[0] == 2:
#             labels = labels - 1
#         return {'window': window, 
#                 'labels': labels}

class EEGDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.csv_file = pd.read_csv(csv_file, delimiter='\t', header=None)
        self.transform = transform
    
    def __len__(self):
        return len(self.csv_file)
    
    def __getitem__(self, idx):
        # if idx >= len(self.csv_file) / 625:
        #     return

        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample_size = 625
        
        # window = self.csv_file.iloc[idx*625:idx*625+625, 1:17].transpose()
        window = self.csv_file.iloc[idx*sample_size:idx*sample_size+sample_size, 1:17].transpose()
        labels = [self.csv_file.iloc[idx*sample_size, 32]]#:idx*625+625, 32] # remove brackets for timestep prediction
        sample = {'window': window, 'labels': labels}

        if self.transform:
            sample = self.transform(sample)
        
        return sample

eeg_dataset = EEGDataset(csv_file='./randomRBFull.csv', transform=transforms.Compose([ToTensor(),
                                                                                  AbsoluteValue(),
                                                                                  MinNormalize()]))

eeg_dataset = torch.utils.data.Subset(eeg_dataset, range(0, 205))

# torch.set_printoptions(profile="full")
torch.set_printoptions(precision=32)
print(eeg_dataset[0]['window'])
torch.set_printoptions(profile="default")