import cv2 as cv
import logging
import numpy as np
import time
import matplotlib.pyplot as plt
import src.tools.util as util
import os
import src.evaluation.ImageBean as ib
class SubEvalIdx:
    def __init__(self, ImageBean):
        self.imWidth = 1024
        self.imHeight = 768
        self.unCorrectedImg = cv.resize(cv.imread(ImageBean.unCorrectedPath), (self.imWidth, self.imHeight))
        self.cmpImg = cv.resize(cv.imread(ImageBean.ourPath), (self.imWidth, self.imHeight))
        self.bimberImg = cv.resize(cv.imread(ImageBean.BimberPath), (self.imWidth, self.imHeight))
        self.CompenNetImg = cv.resize(cv.imread(ImageBean.CompenNetPath),(self.imWidth, self.imHeight))
        self.TPSImg = cv.resize(cv.imread(ImageBean.TpsPath), (self.imWidth, self.imHeight))
        self.srcImg = cv.resize(cv.imread(ImageBean.srcPath), (self.imWidth, self.imHeight))
        self.saveDir = ImageBean.saveDir
        self.imChannel = 3
        self.logger = logging.getLogger("SubEvalLog")
        self.logger.setLevel(logging.DEBUG)
        self.gap = 10

    def Get3ChannelHist(self, image_3chanal):
        # 按R、G、B三个通道分别计算颜色直方图
        image_3chanal = cv.resize(image_3chanal,(self.imWidth, self.imHeight))
        b_hist = cv.calcHist([image_3chanal], [0], None, [256], [0, 256])
        g_hist = cv.calcHist([image_3chanal], [1], None, [256], [0, 256])
        r_hist = cv.calcHist([image_3chanal], [2], None, [256], [0, 256])
        return (b_hist+g_hist+r_hist)/3

    def GetBoxLight(self, boxSize = (16,16), img=None):
        arr = np.ones((self.imWidth//boxSize[0]*self.imHeight//boxSize[1]+1, 1), dtype=np.float32)
        idx = 0
        for i in range(0, self.imWidth//boxSize[0]):
            for j in range(0, self.imHeight//boxSize[1]):
                box = np.sum(img[i*boxSize[0]:(i+1)*boxSize[0], j*boxSize[0]:(j+1)*boxSize[1]])/3
                arr[idx] = np.sum(box)/(boxSize[0]*boxSize[1])
                idx += 1
        return arr

    def CheckImage(self, img):
        if img is None:
            self.logger.debug("image can not be None!")
            exit(0)

    def SubImage(self, srcImg, cmpImg):
        self.CheckImage(srcImg)
        self.CheckImage(cmpImg)
        return cv.subtract(srcImg, cmpImg)

    def ShowImage(self, fileName, img):
        self.CheckImage(img)
        cv.imshow(fileName, img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def Resize(self, imgs):
        ret = []
        for img in imgs:
            self.CheckImage(img)
            ret.append(cv.resize(img, (self.imWidth, self.imHeight)))
        return ret


    def SaveImages(self, srcImg, cmpImg, subImg):
        self.CheckImage(srcImg)
        self.CheckImage(cmpImg)
        self.CheckImage(subImg)
        srcImg, cmpImg, subImg = self.Resize([srcImg, cmpImg, subImg])
        newImg = np.zeros([ self.imHeight+2*self.gap,3*self.imWidth+4*self.gap, self.imChannel], np.uint8)
        newImg.fill(255)
        newImg[self.gap:self.imHeight+self.gap, self.gap:self.imWidth+self.gap, :] = srcImg
        newImg[self.gap:self.imHeight+self.gap, self.imWidth+2*self.gap:self.imWidth*2+2*self.gap, :] = cmpImg
        newImg[self.gap:self.imHeight+self.gap, self.imWidth*2+3*self.gap:self.imWidth*3+3*self.gap, :] = subImg
        cv.imwrite(self.saveDir+time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) +".jpg", newImg)

    def SaveImages2(self, imgs):
        n = len(imgs)
        imgs = self.Resize(imgs)
        newImg = np.zeros([ self.imHeight+2*self.gap,n*self.imWidth+(n+1)*self.gap, self.imChannel], np.uint8)
        newImg.fill(255)
        for i in range(1, n+1):
            img = imgs[i-1]
            self.CheckImage(img)
            newImg[self.gap:self.imHeight + self.gap, self.gap*i+self.imWidth*(i-1):self.imWidth*i + i*self.gap, :] = img
        cv.imwrite(self.saveDir + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + ".jpg", newImg)

    def GetIndex(self):
        subImg = self.SubImage(self.unCorrectedImg, self.cmpImg)
        # self.SaveImages(self.srcImg, self.cmpImg, subImg)
        histo = self.hsv_histogram(subImg)
        print(histo.shape)
        self.logger.info("Get Index Image Done!")
        newImg = util.CombineImages1D([self.unCorrectedImg, self.cmpImg, subImg, cv.resize(histo, (self.imWidth, self.imHeight))])
        return newImg

    def GetIndexWithGT(self):
        gtImg = self.srcImg
        gtImg = cv.resize(gtImg, (self.imWidth, self.imHeight))
        ourSubImg = self.SubImage(self.unCorrectedImg, self.cmpImg)
        BimberSubImg = self.SubImage(self.unCorrectedImg, self.bimberImg)
        CompenNetSubImg = self.SubImage(self.unCorrectedImg, self.CompenNetImg)
        # histo = self.hsv_histogram(subImg)
        newImg = util.CombineImages1DXLim([gtImg, self.unCorrectedImg, self.cmpImg, ourSubImg, self.bimberImg,
                                           BimberSubImg,self.CompenNetImg, CompenNetSubImg])
        self.logger.info("Get Index With GT Image Done!")
        return newImg

    def hsv_histogram(self, image_3chanal):
        # 按R、G、B三个通道分别计算颜色直方图
        b_hist = cv.calcHist([image_3chanal], [0], None, [256], [0, 256])
        g_hist = cv.calcHist([image_3chanal], [1], None, [256], [0, 256])
        r_hist = cv.calcHist([image_3chanal], [2], None, [256], [0, 256])
        # 显示3个通道的颜色直方图
        plt.plot(b_hist, label='B', color='blue')
        plt.plot(g_hist, label='G', color='green')
        plt.plot(r_hist, label='R', color='red')
        plt.legend(loc='best')
        plt.xlim([0, 100])
        plt.grid()
        plt.savefig("./temp.jpg")
        plt.close()
        img = cv.imread("./temp.jpg")
        return img

    '''
    using correct image RGB value / uncorrected RGB value.
    '''
    def PercentageIndex(self, prjImg = None, cmpImg = None):
        if prjImg is None and cmpImg is None:
            prjImg = self.unCorrectedImg
            cmpImg = self.cmpImg
        prjTotal = np.sum(prjImg)
        cmpTotal = np.sum(cmpImg)
        return 1 - cmpTotal/prjTotal

    def PercentageIndeies(self):
        a = self.PercentageIndex(self.unCorrectedImg, self.cmpImg)
        b = self.PercentageIndex(self.unCorrectedImg, self.bimberImg)
        c = self.PercentageIndex(self.unCorrectedImg, self.TPSImg)
        d = self.PercentageIndex(self.unCorrectedImg, self.CompenNetImg)
        return [a, b, c, d]

    def CompareRGBSum(self):
        src_hist = self.Get3ChannelHist(self.unCorrectedImg)
        our_hist = self.Get3ChannelHist(self.cmpImg)
        bimber_hist = self.Get3ChannelHist(self.bimberImg)
        compen_hist = self.Get3ChannelHist(self.CompenNetImg)
        tps_hist = self.Get3ChannelHist(self.TPSImg)

        plt.plot(src_hist, label='unCompensated', color='blue')
        plt.plot(our_hist, label='Our', color='red')
        plt.plot(bimber_hist, label='O.Bimber', color='green')
        plt.plot(compen_hist, label='CompenNet', color='yellow')
        plt.plot(tps_hist, label='TPS', color='black')
        plt.legend(loc='best')
        plt.xlim([0, 256])
        plt.grid()
        plt.savefig("./temp.jpg")
        img = cv.imread("./temp.jpg")
        plt.close()
        return img

def GetImagesRGBSum(imIdx):
    beans = []
    for i in imIdx:
        imBean = ib.ImageBean(srcPath="C:\canary\data\desire\\"+str(i)+".jpg",
                              ourPath="C:\canary\data\\2019.8.11\Ours\\"+str(i)+".jpg",
                              unCorrectedPath="C:\canary\data\\2019.8.11\Groundtruth\\"+str(i)+".jpg",
                              CompenNetPath="C:\canary\data\\2019.8.11\CompenNet\\"+str(i)+".jpg",
                              TpsPath="C:\canary\data\\2019.8.11\Tps\\"+str(i)+".jpg",
                              BimberPath="C:\canary\data\\2019.8.11\Scatter\\"+str(i)+".jpg",
                              saveDir="./")
        beans.append(imBean)
    RGBSums = []
    for bean in beans:
        obj = SubEvalIdx(bean)
        img = obj.CompareRGBSum()
        RGBSums.append(img)
    img = util.CombineImages1D(RGBSums)
    cv.imwrite("./RGBSums.jpg", img)

def GetImagePercentageIndex(imIdx):
    beans = []
    for i in imIdx:
        imBean = ib.ImageBean(srcPath="C:\canary\data\desire\\" + str(i) + ".jpg",
                              ourPath="C:\canary\data\\2019.8.11\Ours\\" + str(i) + ".jpg",
                              unCorrectedPath="C:\canary\data\\2019.8.11\Groundtruth\\" + str(i) + ".jpg",
                              CompenNetPath="C:\canary\data\\2019.8.11\CompenNet\\" + str(i) + ".jpg",
                              TpsPath="C:\canary\data\\2019.8.11\Tps\\" + str(i) + ".jpg",
                              BimberPath="C:\canary\data\\2019.8.11\Scatter\\" + str(i) + ".jpg",
                              saveDir="./")
        beans.append(imBean)
    indeies = []
    for bean in beans:
        obj = SubEvalIdx(bean)
        idx = obj.PercentageIndeies()
        indeies.append(idx)
    indeies = np.array(indeies)
    indeies = np.sum(indeies, axis=0)/len(indeies)
    print("Ours:{:<}".format(indeies[0]))
    print("Bimber:{:<}".format(indeies[1]))
    print("Tps:{:<}".format(indeies[2]))
    print("CompenNet:{:<}".format(indeies[3]))

def GetSubEval(imIdx):
    beans = []
    for i in imIdx:
        imBean = ib.ImageBean(srcPath="C:\canary\data\desire\\" + str(i) + ".jpg",
                              ourPath="C:\canary\data\\2019.8.11\Ours\\" + str(i) + ".jpg",
                              unCorrectedPath="C:\canary\data\\2019.8.11\Groundtruth\\" + str(i) + ".jpg",
                              CompenNetPath="C:\canary\data\\2019.8.11\CompenNet\\" + str(i) + ".jpg",
                              TpsPath="C:\canary\data\\2019.8.11\Tps\\" + str(i) + ".jpg",
                              BimberPath="C:\canary\data\\2019.8.11\Scatter\\" + str(i) + ".jpg",
                              saveDir="./")
        beans.append(imBean)
    newImgs = []
    for bean in beans:
        obj = SubEvalIdx(bean)
        idx = obj.GetIndexWithGT()
        newImgs.append(idx)
    img = util.CombineImages1DYLim(newImgs)
    cv.imwrite("./SubEval.jpg", img)
    pass

if __name__ == '__main__':
    # GetImagesRGBSum([1, 17, 11, 16])
    # GetImagePercentageIndex([1, 17, 11, 16])
    GetSubEval([1, 17, 11, 16])