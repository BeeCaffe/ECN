from torch.utils.data import Dataset
import os
import cv2 as cv
"""
:brief:This class extends from Dataset and implements '__getitem__()' and '__len__' function to be a Dataset
        which can be passed to the 'Dataloader' and produce a dataset 
"""

class MyDataset(Dataset):
    def __init__(self, data_root, index=None, size=None):
        self.data_root = data_root
        self.size = size
        img_list = sorted(os.listdir(data_root))
        img_list = [img_list[x] for x in index] if index is not None else img_list
        self.img_names = [os.path.join(self.data_root, name) for name in img_list]

    def __getitem__(self, index):
        if self.size is not None:
            im = cv.cvtColor(cv.resize(cv.imread(self.img_names[index]), self.size[::-1]), cv.COLOR_BGR2RGB)
        else:
            im = cv.cvtColor(cv.imread(self.img_names[index]), cv.COLOR_BGR2RGB)
        return im
    def __len__(self):
        return len(self.img_names)

