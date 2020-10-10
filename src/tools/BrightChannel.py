import cv2 as cv
import numpy as np
import src.tools.util as utils
import time
import src.Beans.BrightChannelBean as Bean

class BrightChannel:
    def __init__(self, BrightChannelBean):
        self.width = 1024
        self.height = 768
        self.imgPath = BrightChannelBean.imgPath
        self.saveDir = BrightChannelBean.saveDir
        self.img = cv.resize(cv.imread(self.imgPath), (self.width, self.height))

    def GetBrightChannel(self):
        brightChannel = np.max(self.img, 2)
        brightChannel = cv.cvtColor(brightChannel, cv.COLOR_GRAY2BGR)
        return brightChannel

    def GetBrightChannelWithBinary(self):
        brightChannel = np.max(self.img, 2)
        mean = brightChannel.sum() / (self.width * self.height)
        _, brightChannel = cv.threshold(brightChannel, 100, 255, cv.THRESH_BINARY)
        brightChannel = cv.cvtColor(brightChannel, cv.COLOR_GRAY2BGR)
        return brightChannel

    def SaveBrightChannel(self):
        img = utils.CombineImages1DXLim([self.img, self.GetBrightChannel()])
        cv.imwrite(self.saveDir+time.strftime("%Y_%m_%d_%H_%M_%S_bright", time.localtime()) +".jpg", img)
        img = utils.CombineImages1DXLim([self.img, self.GetBrightChannelWithBinary()])
        cv.imwrite(self.saveDir + time.strftime("%Y_%m_%d_%H_%M_%S_mask", time.localtime()) + ".jpg", img)

if __name__=="__main__":
    bean = Bean.BrightChannelBean("res/BrightChannelImages/Camera/308_c.jpg")
    bc = BrightChannel(bean)
    bc.SaveBrightChannel()
    pass
