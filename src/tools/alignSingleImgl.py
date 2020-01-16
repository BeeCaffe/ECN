import cv2
import os
import numpy as np
import util


def alignImages(SrcFilePath,SavePath):
    """
        :brief aligns all images in the file path of SrcFilePath and save to savepath,if savepath notexist,
                it will build a file.meanwhile,it will split the single image into two part ,and align left
                part and right part respectively.
    """
    fileNameList=os.listdir(SrcFilePath)
    #if Save path not exist, build it
    if not os.path.exists(SavePath):
        os.mkdir(SavePath)
    #travel all images
    for i,filename in enumerate(fileNameList):
        #read a image
        img=cv2.imread(SrcFilePath+filename,cv2.IMREAD_UNCHANGED)
        img= util.Resize(img, util.IMG_WIDTH, util.IMG_HEIGHT)
        #split it to two part
        imgLeft,imgRight= util.SplitImageByLine(img, 478)
        #use h_left to correct left part,use h_right to correct right part
        h_left=np.loadtxt("./h_left.txt",dtype=np.float32)
        h_right=np.loadtxt("./h_right.txt",dtype=np.float32)
        imgLeft = util.AlignImageWithH(imgLeft, h_left)
        imgRight = util.AlignImageWithH(imgRight, h_right)
        #combine two half images
        img= util.CombineImages(imgLeft, imgRight)
        # prj=util.FillGap(prj)
        #save image to the file
        cv2.imwrite(SavePath+filename,img)
        print("aligned %d images"%i)

def getHMatrix(imgPath,refImgPath):
        # goal:
        #         get the h_left and h_right matrix using the chessboard image
        # args:
        #         imgPath-the chessboard which is going to be aligned
        #         refImgPath-the unified chessboard,which as the reference to get the homography images
        # return;
        #         null
        #read image
        img=cv2.imread(imgPath,cv2.IMREAD_UNCHANGED)
        refImg=cv2.imread(refImgPath,cv2.IMREAD_UNCHANGED)
        #split the image into left and right part
        imgLeft,imgRight= util.SplitImage(img)
        refImgLeft,refImgRight= util.SplitImageByLine(refImg, 478)
        #use the images to get the h_left and h-right matrix
        regImg,h_left= util.AlignImage(imgLeft, refImgLeft)
        refImg,h_right= util.AlignImage(imgRight, refImgRight)
        #store the homography matrix
        np.savetxt("./h_left.txt",h_left)
        np.savetxt("./h_right.txt",h_right)
        print("successfully generate the homography matrix")

def fillGap(gap,filePath,savePath):
    """
    :brief : the corrected images always exist gap in the middle of image ,I try to move a step of "gap" the right
    part of images to the left,and save the filled images.
    :gap : the gap length which you want to move
    :filepath : the image path
    :savepath : save path
    """
    if not os.path.exists(savePath):
        os.mkdir(savePath)
    name_list = os.listdir(filePath)
    name_list.sort(key=lambda x: int(x[:-4]))
    for i, name in enumerate(name_list):
        imgPath = filePath+os.sep+name
        img = cv2.imread(imgPath,cv2.IMREAD_UNCHANGED)
        height, width, channels = img.shape
        mid = int(width/2)
        n_img = np.zeros((height, width-gap, channels), dtype=np.uint8)
        n_img[0:height, 0:mid] = img[0:height, 0:mid]
        n_img[0:height, mid:(width-gap)] = img[0:height, (mid+gap): width]
        n_img = n_img[0:height, 0:(width-gap)]
        n_img = util.Resize(n_img, 1024, 768)
        cv2.imwrite(savepath+name, n_img)
        print("have dealed {:<d} images".format(i))

if __name__=="__main__":
    # #get the h_left and h_right matrix
    # refPath = "C:\\Users\\Administrator\\PycharmProjects\\RadianceComponet\\resources\\image6.4_prj\\00005.jpg"
    # imgPath = "C:\\Users\\Administrator\\PycharmProjects\\RadianceComponet\\resources\\images6.4_cam_src\\00005.jpg"
    getHMatrix(imgPath,refPath)
    # #align all images and save it to savepath
    # srcPath = "C:\\Users\\Administrator\\PycharmProjects\\RadianceComponet\\resources\\images6.4_cam_src\\"
    # savePath = "C:\\Users\\Administrator\\PycharmProjects\\RadianceComponet\\resources\\images6.4_cam_geocor\\"
    # alignImages(srcPath,savePath)
    filepath = "C:\\Users\\Administrator\\PycharmProjects\\RadianceComponet\\resources\\images6.4_cam_geocor\\"
    savepath = "C:\\Users\\Administrator\\PycharmProjects\\RadianceComponet\\resources\\images6.4_cam_filledgap\\"
    fillGap(32,filepath, savepath)
