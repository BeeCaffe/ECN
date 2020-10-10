import numpy as np
import cv2 as cv
import os
import time
import math
import src.tools.util as utils
class Bimber:
    def __init__(self, imRoot, saveRoot="./"):
        self.width = 1024
        self.height = 768
        self.screenWidth = 0.47
        self.screenHeight = 0.18
        self.screenMidHeight = 0.255
        self.centerDist = 0.55
        self.patchSize = 16
        self.imRoot = imRoot
        self.saveRoot = saveRoot
        self.u = self.width//self.patchSize
        self.v = self.height//self.patchSize
        self.detaX = self.screenWidth/self.u
        self.detaYEd = self.screenHeight/self.v
        self.detaYEd = self.screenMidHeight/self.v
        self.PI = 3.14159265358979323846
        self.f = 1
        self.onePathchF()
        self.doublePatchF()

    def onePathchF(self):
        self.Fi = np.zeros((self.v, self.u), np.float32)
        for i in range(self.u*self.v):
            xi = (i%self.u)*self.detaX+self.detaX/2.
            yi = self.screenHeight - (i/self.u)*self.detaYEd/2.
            d = math.sqrt(-(1.1 * (self.screenWidth / 2. - xi) * 0.5253 - (self.screenWidth / 2. - xi) * (self.screenWidth / 2. - xi) - math.pow(.55, 2)));
            ri = math.sqrt(d * d + yi * yi)
            alpha_i = self.PI / 2 - math.atan(d / yi)
            dA = 1. / (self.u * self.v)
            ftmp = dA * math.cos(alpha_i) / (ri * ri * self.PI)
            col = int(i % self.u)
            row = i // self.u
            self.Fi[row, col] = self.f * ftmp

    def doublePatchF(self):
        self.Fij = np.zeros((self.u * self.v, self.u * self.v), dtype=np.float32)
        for i in range(self.u*self.v):
            for j in range(self.u*self.v):
                if i == j:
                    self.Fij[i, j] = 0.
                else:
                    if (i % self.u <= self.u / 2) and (j % self.u >= self.u / 2):
                        di = self.screenWidth / 2-(i % self.u) * self.detaX-self.detaX / 2.
                        dj = (j % self.u-self.u / 2) * self.detaX+self.detaX / 2
                        rij = math.sqrt(di * di+dj * dj)
                        beta_i = math.atan(dj / di)
                        beta_j = math.atan(di / dj)
                        alpha_i = self.PI / 2-beta_i
                        alpha_j = self.PI / 2-beta_j
                        dA = 1. / (self.u * self.v)
                        ftmp = dA * math.cos(alpha_i) * math.cos(alpha_j) / (rij * rij * self.PI)
                        self.Fij[i, j] = ftmp
                    elif (i % self.u >= self.u / 2) and (j % self.u <= self.u / 2):
                        dj=self.screenWidth / 2-(j % self.u) * self.detaX-self.detaX / 2.
                        di=(i % self.u-self.u / 2) * self.detaX+self.detaX / 2
                        rij=math.sqrt(di * di+dj * dj)
                        beta_i=math.atan(dj / di)
                        beta_j=math.atan(di / dj)
                        alpha_i=self.PI / 2-beta_i
                        alpha_j=self.PI / 2-beta_j
                        dA=1. / (self.u *self.v)
                        ftmp=dA * math.cos(alpha_i) * math.cos(alpha_j) / (rij * rij * self.PI);
                        self.Fij[i, j]=ftmp
                    else:
                        self.Fij[i, j]=0
        for i in range(self.u*self.v):
            sm = np.sum(self.Fij[i, :])
            self.Fij[i, :] /= sm

    def computeScatter(self, I):
        S = np.zeros((self.height, self.width, 3),dtype=np.float32)
        stTime = time.time()
        print("Computing S : ")
        for i in range(0, self.u*self.v):
            row = i//self.u
            col = i%self.u
            patch = I[row*self.patchSize:(row+1)*self.patchSize, col*self.patchSize:(col+1)*self.patchSize]
            for j in range(0, self.u*self.v):
                S[row*self.patchSize:(row+1)*self.patchSize, col*self.patchSize:(col+1)*self.patchSize] += patch*self.Fi[row, col]*self.Fij[j, i]
                endTime = time.time()
                utils.process("Computing S", i*j, self.u*self.v*self.u*self.v, stTime, endTime)
        return S

    '''
    R: the initial image
    S: the compensated image
    '''
    def CompensateI(self, R, S):
        I = np.array((self.height, self.width, 3), dtype=np.float32)
        cols = np.size(R, 1)
        rows = np.size(R, 0)
        stTime = time.time()
        print("Compensating : ")
        I = np.subtract(R, S)
        for i in range(self.u*self.v):
            row = i//self.u
            col = i%self.u
            I[row*self.patchSize:(row+1)*self.patchSize, col*self.patchSize:(col+1)*self.patchSize]/=self.Fi[i]
            endTime = time.time()
            utils.process("Compensating", i, self.u*self.v, stTime,endTime)
        return I

    def compensateImg(self,R):
        S = self.computeScatter(R)
        I = self.CompensateI(R,S)
        S_next = self.computeScatter(I)
        return R-S_next

    def compensateImgs(self):
        nameLists = os.listdir(self.imRoot)
        for name in nameLists:
            imgPath = self.imRoot+"/"+name
            img = cv.resize(cv.imread(imgPath), (self.width, self.height))
            img = np.array(img,dtype=np.float32)
            n_img = self.compensateImg(img)
            saveName = self.saveRoot+"/"+name
            cv.imwrite(saveName,n_img)
            print("compensated an image")
        print('Done!')

if __name__=='__main__':
    bimber = Bimber(imRoot="C:\canary\data\desire")
    bimber.compensateImgs()