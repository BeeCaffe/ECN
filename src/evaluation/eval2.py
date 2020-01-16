from src.tools.util import *
import splitTo256 as sp
import cv2 as cv
import time
from queue import Queue

def splits():
    Root = 'resources/evaluation/'
    Name = ['CompenNet/', 'ECN/', 'GroundTruth/', 'Scatter', 'TPS']

    for name in Name:
        filePath = Root+name
        savePath = Root+name[:-1]+'256/'
        sp.split(data_root=filePath, save_root=savePath, maxNums=10000)
    print('Done!')

def eval():

    Root = 'resources/evaluation/'
    Name = ['GroundTruth256/','CompenNet256/', 'Enhanced-CompenNet256/', 'Scatter256/', 'TPS256/']
    groundTruth = Root+'GroundTruth256/'

    groundTruthList = os.listdir(groundTruth)
    groundTruthList.sort(key=lambda x:int(x[:-4]))

    for name in Name:
        path = Root+name
        pathList = os.listdir(path)
        pathList.sort(key=lambda x:int(x[:-4]))
        psnrList = []
        ssimList = []
        mseList = []
        startTm=time.time()
        for i, imgname in enumerate(pathList):
            imgPath = path+imgname
            refPath = groundTruth+imgname
            img = cv.imread(imgPath, cv.IMREAD_UNCHANGED)
            imgRef = cv.imread(refPath, cv.IMREAD_UNCHANGED)
            # cv.imshow("img1", img)
            # cv.imshow('img2', imgRef)
            # cv.waitKey(0)
            # cv.destroyAllWindows()
            psnrList.append(psnr(img, imgRef))
            ssimList.append(ssim(img, imgRef))
            mseList.append(mse(img, imgRef))
            nowTm=time.time()
            process(name[:-1], i, len(pathList), startTm, nowTm)

        with open('test/evaluate/'+name[:-1]+'.txt', 'w') as f:
            f.write('{:<10}{:<10}{:<10}\n'.format('SSIM', 'PSNR', 'RMSE'))
            mean = [0., 0., 0.]
            for myssim, mypsnr, mymse in zip(psnrList,ssimList,mseList):
                f.write('{:<2.4f}   {:<2.4f}    {:<2.4f}\n'.format(myssim, mypsnr, mymse))
                mean[0]+=myssim/len(pathList)
                mean[1]+=mypsnr/len(pathList)
                mean[2]+=mymse/len(pathList)

            f.write('{:<10}{:<10}{:<10}\n'.format('MSSIM', 'MPSNR', 'MRMSE'))
            f.write('{:<2.4f}   {:<2.4f}    {:<2.4f}\n'.format(mean[0], mean[1], mean[2]))
if __name__=='__main__':
    # splits()
    eval()