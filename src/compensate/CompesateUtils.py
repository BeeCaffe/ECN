import cv2 as cv
import torch
import os
import numpy as np
import ECN
def compensate(prjRoot=None, surfPath=None, pthPath=None, saveRoot=None, MaxNum=2):
    """
    :brief : this method is used to compensate the GroundTruth images using trained weights,which saves in xxx.pth
            file.you only need pass the GroundTruth images root file , the surface image path, xxx.pth filepath
            save root file,and the max number of you suppose to compensate
    :param prjRoot: GroundTruth images's root file
    :param surfPath: surface image's path
    :param pthPath: pth file's path
    :param saveRoot: save root file
    :param MaxNum: max numbers of you suppose to compensate
    :return: null
    """
    if not os.path.exists(saveRoot):
        os.makedirs(saveRoot)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #surface
    surfImg = cv.cvtColor(cv.imread(surfPath), cv.COLOR_BGR2RGB)
    h,w,ch = surfImg.shape
    # surfImg = torch.from_numpy(surfImg).expand([1, 768, 1024, 3])
    surfImg = torch.from_numpy(surfImg).expand([1, h, w, ch])
    surfImg = surfImg.permute((0, 3, 1, 2)).float().div(255)
    #GroundTruth
    nameList = os.listdir(prjRoot)
    nameList.sort(key=lambda x: int(x[:-4]))
    #main
    for i, name in enumerate(nameList):
        # read each picture and feed it to CompeNet Model
        prj_img = cv.cvtColor(cv.imread(prjRoot + name), cv.COLOR_BGR2RGB)
        h,w,ch = prj_img.shape
        # prj_img = torch.from_numpy(prj_img).expand([1, 768, 1024, 3])
        prj_img = torch.from_numpy(prj_img).expand([1, h, w, ch])
        prj_img = prj_img.permute((0, 3, 1, 2)).float().div(255)
        # load model
        model = ECN.ECN().cuda()
        model.load_state_dict(torch.load(pthPath))
        print(prj_img.shape)
        print(surfImg.shape)
        pred = model(prj_img.to(device), surfImg.to(device))
        # save pred_l1_125_5000 to you desired path
        pred = pred[0, :, :, :]
        pred = np.uint8((pred[:, :, :] * 255).permute(1, 2, 0).cpu().detach().numpy())
        pred = pred[:, :, ::-1]

        cv.imwrite(saveRoot + str(i) + '.jpg', pred)
        print("has componeted {:<d} pictures, and compensated pictures saved in {:<s}".format(i, saveRoot))
        if i > MaxNum:
            exit(0)

if __name__=='__main__':
    prjRoot = 'H:\\python\\VGG_CompeNet2\\src\\compensate\\prj\\'
    surfPath = 'H:\\python\\VGG_CompeNet2\\src\\compensate\\surf\\surf.jpg'
    pthPath = 'H:\\python\\VGG_CompeNet2\\src\\compensate\\weight\\CompeNet_5000_3000_2_0.001_0.2_800_l1+l2+ssim+vggLoss.pth'
    saveRoot = 'H:\\python\\VGG_CompeNet2\\src\\compensate\\res\\'
    compensate(prjRoot,surfPath,pthPath,saveRoot,1)
