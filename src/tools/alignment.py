from src.tools.util import *
import cv2 as cv
import time

MidOffset=20
FillGapMidOffset=-10
# subDir = 'Tps/'
# subDir = 'CompenNet/'
# subDir = 'Groundtruth/'
# subDir = 'Ours/'
subDir = 'Ours/'

# getWindowImg = 'resources/eval3/uncorrect/Groundtruth/21.JPG'
getWindowImg = 'resources/eval3/uncorrect/Ours/21.JPG'

needToAlignImgRoot = 'resources/eval3/uncorrect/'+subDir
getWindowSaveRoot = 'resources/eval3/getWindow/'+subDir
alignSaveRoot = 'resources/eval3/correct/'+subDir
# geometric correct image sample
geoCorrectImg = 'resources/eval3/getWindow/Ours/10.JPG'
geoCorrectRefImg='resources/eval3/uncorrect/projector/10.jpg'

def resize(imRoot):
    nameList = os.listdir(imRoot)
    for name in nameList:
        imgPath=imRoot+name
        savePath = 'resources/tmp/img/'+name
        img = cv.imread(imgPath)
        img = Resize(img, width=1024, height=768)
        cv.imwrite(savePath, img)
    print('Done!')

def getROI(imRoot,saveRoot,loc):
    names=os.listdir(imRoot)
    x,y,w,h = loc
    if not os.path.exists(saveRoot):
        os.mkdir(saveRoot)
    for name in names:
        imgPath = imRoot+name
        savePath = saveRoot+name
        img=cv.imread(imgPath)
        img=GetRegion(img, pt=(x, y), width=w, height=h)
        img=cv.resize(img, (1024,768))
        cv.imwrite(savePath,img)
    print('Done!')

def getHMat(refImPair,mid):
    imgLeft, imgRight = SplitImageByLine(refImPair[0], mid)
    refLeft, refRight = SplitImageByLine(refImPair[1], mid)
    imgReg, h_left = AlignImage(imgLeft, refImg=refLeft)
    imgReg, h_right = AlignImage(imgRight, refImg=refRight)
    return h_left, h_right

def alignment(imRoot,saveRoot,h_left,h_right,mid):
    names=os.listdir(imRoot)
    if not os.path.exists(saveRoot):
        os.mkdir(saveRoot)
    stTm=time.time()
    for i, name in enumerate(names):
        imgPath = imRoot+name
        savePath = saveRoot+name
        img=cv.imread(imgPath)
        imgLeft,imgRight=SplitImage(img)
        imgLeft=AlignImageWithH(imgLeft,h_left)
        imgRight=AlignImageWithH(imgRight,h_right)
        img=CombineImages(imgLeft,imgRight)
        img = cv.resize(img, (1024, 768))
        if i==0:
            lineLeft, lineRight = findToLine(img,img.shape[1]//2+FillGapMidOffset)
        img=FillGap(img, lineLeft, lineRight)
        cv.imwrite(savePath, img)
        nowTm=time.time()
        process("aligning :", i, len(names), stTm, nowTm)
    print("Done!")

# align single image, you should set the paths by yourself
if __name__=='__main__':
    #get suit window
    img = cv.imread(getWindowImg)
    x, y, w, h, m = getWindow(img)
    # the camera capture image root
    imRoot=needToAlignImgRoot
    # get window save image root
    saveRoot=getWindowSaveRoot
    # aligned image save root
    alignSaveRoot = alignSaveRoot
    getROI(imRoot, saveRoot, [x, y, w, h])
    #geometric correct image sample
    im = cv.resize(cv.imread(geoCorrectImg),(1024,768))
    #geometric correct reference imag,
    imRef = cv.resize(cv.imread(geoCorrectRefImg),(1024,768))
    refImPair = [im, imRef]
    mid = 1024//2+MidOffset
    cv2.line(im, pt1=(mid, 0), pt2=(mid, im.shape[0]), color=(0, 255, 255), thickness=2)
    cv2.imshow("img", cv2.resize(im, (1024, 768)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    h_left, h_right=getHMat(refImPair, mid)
    alignment(saveRoot,alignSaveRoot, h_left, h_right, mid)

# if __name__=='__main__':
#     img = cv.imread('resources/eval/loss/l1+l2+vgg.JPG')
#     x, y, w, h, m = getWindow(img)
#     imRoot = 'resources/eval/loss/'
#     saveRoot = 'resources/eval/suitLoss/'
#     getROI(imRoot, saveRoot, [x, y, w, h])
#     pass