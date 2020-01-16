import MyDataset
from torch.utils.data import DataLoader
import torch
def LoadTrainAndValid(dataset_root, paths, trainNums):
    """
    :brief : this function is used to make train dataset and valid dataset, where you only need to pass the
            'dataset_root' and the path list 'paths',and it will combine those path and produce train and
            valid dataset.
    :param dataset_root: the root path of you place train and valid data
    :param paths:the sub path of dataset,where including 'cam','prj','surf'
    :param trainNums: train numbers of how many images you suppose to use to train.
    :return train_data, valid_data:each of them are two dictionary ,including 'cam','valid' and 'surf',the
            shape of those two dict is train_data==>[[img_nums,3,256,256],[img_nums,3,256,256],[img_nums,3,256,256]]
            valid_data==>[[img_nums,3,256,256],[img_nums,3,256,256],[img_nums,3,256,256]]
    """
    train_cam_path = dataset_root+paths[0]
    train_prj_path = dataset_root+paths[1]
    train_surf_path = dataset_root+paths[2]

    valid_cam_path = dataset_root+paths[3]
    valid_prj_path = dataset_root+paths[4]
    valid_surf_path = dataset_root+paths[5]

    train_cam = readImgs(train_cam_path)
    train_prj = readImgs(train_prj_path)
    train_surf = readImgs(train_surf_path)

    valid_cam = readImgs(valid_cam_path)
    valid_prj = readImgs(valid_prj_path)
    valid_surf = readImgs(valid_surf_path)

    train_cam_surf = train_surf.expand_as(train_cam)
    valid_cam_surf = valid_surf.expand_as(valid_cam)

    train_data = {
        'cam': train_cam[:trainNums, :, :, :],
        'prj': train_prj[:trainNums, :, :, :],
        'surf': train_cam_surf[:trainNums, :, :, :]
        }

    valid_data = {
        'cam': valid_cam,
        'prj': valid_prj,
        'surf': valid_cam_surf
        }
    return train_data, valid_data


def readImgs(path):
    imgs = MyDataset.MyDataset(path, index=None, size=None)
    dataset = DataLoader(imgs, batch_size=len(imgs), shuffle=False, drop_last=False)
    for i, img in enumerate(dataset):
        return img.permute((0, 3, 1, 2)).float().div(255)

def loadValidDataset(dataset_root, paths):

    valid_cam_path = dataset_root + paths[3]
    valid_prj_path = dataset_root + paths[4]
    valid_surf_path = dataset_root + paths[5]

    valid_cam = readImgs(valid_cam_path)
    valid_prj = readImgs(valid_prj_path)
    valid_surf = readImgs(valid_surf_path)

    valid_cam_surf = valid_surf.expand_as(valid_cam)

    valid_data = {
        'cam': valid_cam,
        'prj': valid_prj,
        'surf': valid_cam_surf
    }
    return valid_data