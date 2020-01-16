import util
import cv2
import os
import numpy as np


"""
@brief this program is used to do the alignment of camare images and the GroundTruth images,
@Global srcPath-the images file which you want to process, the result will be saved in the same file.
"""
PT = (307, 258)
WIDTH = 1185
HEIGHT =1178

def Resize(srcPath,savePath):
    name_list = os.listdir(srcPath)
    name_list.sort(key=lambda x:int(x[:-4]))
    if not os.path.exists(savePath):
        os.mkdir(savePath)
    for i,name in enumerate(name_list):
        img = cv2.imread(srcPath+name,cv2.IMREAD_UNCHANGED)
        img = util.Resize(img, 1024, 768)
        cv2.imwrite(savePath+name,img)
        print("have resized {:<d} pictures".format(i))

def SplitImages(srcPath,savePath):
    """
    :brief: split the images in the middle.
    """
    if not os.path.exists(savePath):
        os.mkdir(savePath)

    nameList=os.listdir(srcPath)

    nameList.sort(key = lambda x:int(x[:-4]))

    for i,name in enumerate(nameList):

        img=cv2.imread(srcPath+name,cv2.IMREAD_UNCHANGED)

        imgLeft,imgRight= util.SplitImage(img)

        cv2.imwrite(savePath+name,imgLeft)

        print("handled %d image!"%i)

def SplitImgAndCombine(srcPath):
    nameList = os.listdir(srcPath)

    nameList.sort(key=lambda x: int(x[:-4]))

    for i, name in enumerate(nameList):
        img = cv2.imread(srcPath + name, cv2.IMREAD_UNCHANGED)

        imgLeft, imgRight = util.SplitImage(img)

        img = util.CombineImages(imgLeft, imgRight)

        cv2.imwrite(srcPath + name, img)

        print("handled %d image!" % i)
    pass

def align(srcPath,savePath):

    nameList=os.listdir(srcPath)

    nameList.sort(key = lambda x:int(x[:-4]))

    if not os.path.exists(savePath):
        os.mkdir(savePath)

    for i,name in enumerate(nameList):

        img=cv2.imread(srcPath+name,cv2.IMREAD_UNCHANGED)

        img= util.Resize(img, util.IMG_WIDTH, util.IMG_HEIGHT)

        h=np.loadtxt("./h.txt",dtype=np.float32)

        img= util.AlignImageWithH(img, h)

        cv2.imwrite(savePath+name,img)

        print("handled %d images"%i)

def getRIO(srcPath):

    nameList=os.listdir(srcPath)

    nameList.sort(key = lambda x:int(x[:-4]))

    for i,name in enumerate(nameList):

        img = cv2.imread(srcPath+name,cv2.IMREAD_UNCHANGED)

        img = util.GetRegion(img, PT, WIDTH, HEIGHT)

        img = util.Resize(img, 1024, 768)

        cv2.imwrite(srcPath+name,img)

        print("handled %d images"%i)

def getH(imgPath,refPath):

    img=cv2.imread(imgPath)

    ref=cv2.imread(refPath)

    img= util.Resize(img, 1024, 768)

    img,h= util.AlignImage(img, ref)

    np.savetxt("./h.txt",h)

    print("saved h matrix in h.txt")

def drawRect(img):
    """
    :brief: this function is used to get the suitable point ,width and height to spilt images.
    :prj : the image
    :return: null
    """
    pt = PT
    pt2 = (int(PT[0]+WIDTH), int(PT[1]+HEIGHT))
    img = cv2.rectangle(img, pt, pt2, color=[255, 0, 255])
    cv2.imwrite("./test.jpg",img)
    print("draw sucessful!")


if __name__ == "__main__":
    # prj = cv2.imread("C:\\Users\\Administrator\\PycharmProjects\\RadianceComponet\\resources6_5\\images6.5_splited\\00001.jpg")
    # drawRect(prj)
    # srcPath ="C:\\Users\\Administrator\\PycharmProjects\\RadianceComponet\\resources6_5\\images6.5_splited\\"
    srcPath = "C:\\Users\\Administrator\\PycharmProjects\\RadianceComponet\\resources6_5\\images6.5_1024x768\\"
    savePath = "C:\\Users\\Administrator\\PycharmProjects\\RadianceComponet\\resources6_5\\images6.5_aligned\\"
    # getRIO(srcPath)
    # srcPath = "C:\\Users\\Administrator\\PycharmProjects\\RadianceComponet\\resources6_5\\images6.5\\"
    # savePath = "C:\\Users\\Administrator\\PycharmProjects\\RadianceComponet\\resources6_5\\images6.5_splited\\"
    # SplitImages(srcPath,savePath)
    # Resize(srcPath,savePath)
    getH("C:\\Users\\Administrator\\PycharmProjects\\RadianceComponet\\resources6_5\\images6.5_1024x768\\00002.jpg",
         "C:\\Users\\Administrator\\PycharmProjects\\RadianceComponet\\resources6_5\\images\\00002.jpg")
    align(srcPath, savePath)
