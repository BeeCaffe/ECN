import cv2 as cv
import logging
import numpy as np
import time
class SubEvalIdx:
    def __init__(self, srcImg, cmpImg, saveDir = "./"):
        self.srcImg = cv.imread(srcImg)
        self.cmpImg = cv.imread(cmpImg)
        self.saveDir = saveDir
        self.imWidth = 1024
        self.imHeight = 768
        self.imChannel = 3
        self.logger = logging.getLogger("SubEvalLog")
        self.logger.setLevel(logging.DEBUG)
        pass

    def CheckImage(self, img):
        if img is None:
            self.logger.debug("image can not be None!")
            exit(0)

    def SubImage(self, srcImg, cmpImg):
        self.CheckImage(srcImg)
        self.CheckImage(cmpImg)
        return cv.subtract(cmpImg, srcImg)

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
        newImg = np.zeros([ self.imHeight,3*self.imWidth, self.imChannel], np.uint8)
        print(srcImg.shape)
        newImg[0:self.imHeight, 0:self.imWidth,  :] = srcImg
        newImg[0:self.imHeight, self.imWidth:self.imWidth*2, :] = cmpImg
        newImg[0:self.imHeight, self.imWidth*2:self.imWidth*3, :] = subImg
        cv.imwrite(self.saveDir+time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) +".jpg", newImg)

    def GetIndex(self):
        subImg = self.SubImage(self.srcImg, self.cmpImg)
        self.SaveImages(self.srcImg, self.cmpImg, subImg)
        self.logger.info("Get Index Image Done!")


if __name__ == '__main__':
    obj = SubEvalIdx(srcImg="res/SubEvalIdx/src.JPG", cmpImg="res/SubEvalIdx/cmp.JPG")
    obj.GetIndex()
    pass