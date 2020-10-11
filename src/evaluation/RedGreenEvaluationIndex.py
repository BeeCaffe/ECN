import cv2 as cv
import logging
import numpy as np
import time
import matplotlib.pyplot as plt
import src.tools.util as util
import src.Beans.ImageBean as ib

class RedBlueEvaluationIndex():
    def __init__(self, ImageBean):
        self.width = ImageBean.width
        self.height = ImageBean.height
        self.unCorrectedImg = cv.resize(cv.imread(ImageBean.unCorrectedPath), (self.width, self.height))
        self.correctedImg = cv.resize(cv.imread(ImageBean.ourPath), (self.width, self.height))
        self.bimberImg = cv.resize(cv.imread(ImageBean.BimberPath), (self.width, self.height))
        self.CompenNetImg = cv.resize(cv.imread(ImageBean.CompenNetPath), (self.width, self.height))
        self.TPSImg = cv.resize(cv.imread(ImageBean.TpsPath), (self.width, self.height))
        self.srcImg = cv.resize(cv.imread(ImageBean.srcPath), (self.width, self.height))
        self.saveDir = ImageBean.saveDir + "/ReadBlue/"
        self.channel = 3
        self.logger = logging.getLogger("SubEvalLog")
        self.logger.setLevel(logging.DEBUG)

    def GetRedBlueIndex(self):
        redBlue = util.MarkRedBlue(self.unCorrectedImg, self.correctedImg)
        return redBlue

if __name__=="__main__":
    pass