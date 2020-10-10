import brisque
import cv2 as cv
import src.evaluation.ImageBean as ib
import numpy as np
brique = brisque.BRISQUE()
class MyBrisque:
    def __init__(self, ImageBean):
        self.width = 1024
        self.height = 768
        self.path = ImageBean
        self.srcImg = cv.resize(cv.imread(ImageBean.srcPath), (self.width, self.height))
        self.ourImg = cv.resize(cv.imread(ImageBean.ourPath), (self.width, self.height))
        self.unCorrectedImg = cv.resize(cv.imread(ImageBean.unCorrectedPath), (self.width, self.height))
        self.TpsImg = cv.resize(cv.imread(ImageBean.TpsPath), (self.width, self.height))
        self.BimberImg = cv.resize(cv.imread(ImageBean.BimberPath), (self.width, self.height))
        self.CompenNetImg = cv.resize(cv.imread(ImageBean.CompenNetPath), (self.width, self.height))
        self.saveDir = ImageBean.saveDir

        pass

    def getSaturation(self, img):
        img = cv.cvtColor(img, cv.COLOR_BGR2HLS_FULL)
        sat = np.mean(img[2])
        return sat

    def get_our_tps_compen_bimber_saturation(self):
        imgs = [self.TpsImg, self.BimberImg, self.CompenNetImg, self.ourImg ]
        sat = []
        for img in imgs:
            sat.append(self.getSaturation(img))
        return sat

    def getScores(self, imgs):
        '''
        @parma: imgs -> image lists, e.g. [srcimg,ourimage,uncorrectedimage,...]
        '''
        brisques = []
        for img in imgs:
            brisques.append(brique.get_score(img=img))
        return brisques

    def get_our_compenet_tps_bimber_score(self):
        return self.getScores([self.TpsImg, self.BimberImg, self.CompenNetImg, self.ourImg])

def GetBrique(imIdx):
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
    scores = []
    sats = []
    for bean in beans:
        obj = MyBrisque(bean)
        score = obj.get_our_compenet_tps_bimber_score()
        sats.append(obj.get_our_tps_compen_bimber_saturation())
        scores.append(score)
    scores = np.array(scores)
    scores = np.mean(scores, axis=0)
    sats = np.mean(sats, axis=0)
    print("brique score:")
    print(scores)
    print("saturation:")
    print(sats)


if __name__ == '__main__':
    GetBrique([1, 2, 3, 4, 5, 6, 7, 8, 9])