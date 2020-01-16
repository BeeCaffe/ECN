import os
import splitTo256 as sp
import cv2 as cv
import shutil
import util

def makeCompeNetDataset(camDataRoot, prjDataRoot, datasetRoot,validSetNums=400,DatasetNums = 5000):
    """
    :brief: this program is used to make the CompoNet's dataset, before use it, you should set the data_root and the
            dataset,and the DatasetNums which limits the max images number in your dataset
    :return: null
    """
    dataset_list = ["train\\cam\\",
                    "train\\prj\\",
                    "train\\surf\\",
                    "valid\\cam\\",
                    "valid\\prj\\",
                    "valid\\surf\\",
                    "cmp\\cam\\",
                    "cmp\\prj\\",
                    "cmp\\surf\\"]
    for name in dataset_list:
        if not os.path.exists(datasetRoot+name):
            os.makedirs(datasetRoot+name)

    cam_save_path = datasetRoot+dataset_list[0]
    prj_save_path = datasetRoot+dataset_list[1]

    sp.split(camDataRoot, cam_save_path, DatasetNums)
    sp.split(prjDataRoot, prj_save_path, DatasetNums)

    cam_name_list = os.listdir(cam_save_path)
    prj_name_list = os.listdir(prj_save_path)
    cam_name_list.sort(key=lambda x: int(x[:-4]))
    prj_name_list.sort(key=lambda x: int(x[:-4]))

    count = 0
    for cam_name, prj_name in zip(cam_name_list, prj_name_list):
        if count <= validSetNums:
            src_cam_path = cam_save_path + cam_name
            dst_cam_path = dataSetRoot + dataset_list[3]

            src_prj_path = prj_save_path + prj_name
            dst_prj_path = dataSetRoot + dataset_list[4]

            print(src_cam_path, dst_cam_path)
            shutil.move(src_cam_path, dst_cam_path)
            shutil.move(src_prj_path, dst_prj_path)
            count += 1
            print("moving {:<d} images to valid dataset".format(count))
    util.rename(cam_save_path)
    util.rename(prj_save_path)
    print("All Things Done, there are {:<d} images for train, and {:<d} images for test".format(DatasetNums,
                                                                                                    validSetNums))

def makeVggLossDataset():
    DatasetNums = 1000
    ValidSetNums = 200
    data_root = "H:\\python\\VGG_CompeNet\\vgg_pretrain_dataset"
    dataset = ""
    dateset_list =[dataset+"\\train\\pred\\",
                   dataset+"\\train\\prj\\",
                   dataset+"\\valid\\pred\\",
                   dataset+"\\valid\\prj\\"]
    if not os.path.exists(data_root+dataset):
        os.mkdir(data_root+dataset)

    for name in dateset_list:
        path = data_root+dataset+name
        if not os.path.exists(path):
            os.makedirs(path)

    pred_data_path = data_root+"\\prj\\pred_l1_125_5000\\"
    prj_data_path = "H:\\python\\VGG_CompeNet\\resources6_5\\Images\\"

    pred_save_path = data_root+dataset+dateset_list[0]
    prj_save_path =  data_root+dataset+dateset_list[1]

    sp.split(pred_data_path, pred_save_path, DatasetNums)
    sp.split(prj_data_path, prj_save_path, DatasetNums)

    pred_name_list = os.listdir(pred_save_path)
    prj_name_list = os.listdir(prj_save_path)
    pred_name_list.sort(key=lambda x:int(x[:-4]))
    prj_name_list.sort(key=lambda x:int(x[:-4]))

    count = 0

    for pre_name, prj_name in zip(pred_name_list, prj_name_list):
        if count<ValidSetNums:
            src_pred_path = data_root+dataset+dateset_list[0]+pre_name
            dst_pred_path = data_root+dataset+dateset_list[2]

            src_prj_path = data_root+dataset+dateset_list[1]+prj_name
            dst_prj_path = data_root+dataset+dateset_list[3]

            shutil.move(src_pred_path, dst_pred_path)
            shutil.move(src_prj_path, dst_prj_path)
            count+=1
            print("moving {:<d} images to valid dataset".format(count))
    print("All Things Done, there are {:<d} images for train, and {:<d} images for test".format(DatasetNums,ValidSetNums))

if __name__=='__main__':
    # makeVggLossDataset()
    camDataRoot = 'resources/evaluation/aligned/Groundtruth/'
    prjDataRoot = 'resources/evaluation/projector/'
    dataSetRoot = 'resources/eavaluateDataset/Groundtruth/'
    validNums = 300
    makeCompeNetDataset(camDataRoot, prjDataRoot, dataSetRoot, validNums, 6000)


