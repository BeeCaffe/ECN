from src.tools.util import *
import cv2 as cv
from queue import Queue

def computeH():
    """computes H matrix"""
    img = cv2.imread("resources/tmp/tailed/98.jpg")
    refImg = cv2.imread("test/evaluate/00002.jpg")
    imgLeft, imgRight = SplitImageByLine(img, 5)

    refLef, refRight = SplitImage(refImg)
    imgRegLef, hleft=AlignImage(imgLeft, refImg=refLef)
    np.savetxt('./hleft.txt', hleft)

    imgRegRig, hright=AlignImage(imgRight, refImg=refRight)
    np.savetxt('./hright.txt', hright)
    cv2.imshow("img", imgRegRig)
    cv2.waitKey(0)
    cv2.imshow("img", imgRegLef)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def tailer():
    untailRot = 'resources/evaluation/ROI_Eval/ECN/'
    tailedRoot = 'resources/evaluation/ROI_Eval_tailed/ECN/'
    if not os.path.exists(tailedRoot):
        os.mkdir(tailedRoot)
    pt = (650, 160)
    width = 1780
    height = 1600
    namelist = sorted(os.listdir(untailRot))
    for imName in namelist:
        imPath = untailRot+imName
        savePath = tailedRoot+imName
        img = cv.imread(imPath)
        img = GetRegion(img, pt, width=width, height=height)
        img = cv.resize(img, (1024, 768))
        # cv.imshow('img', img)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        cv.imwrite(savePath, img)
    print('Done!')

def alignAllImg():
    imgRoot = 'resources/tmp/tailed/'
    alignedRoot = 'resources/tmp/aligned/'
    hleft = np.loadtxt('./hleft.txt')
    hright = np.loadtxt('./hright.txt')

    namelist = sorted(os.listdir(imgRoot))

    for name in namelist:
        imgPath = imgRoot+name
        savePath = alignedRoot+name
        img = cv.imread(imgPath)
        imgLeft, imgRight = SplitImageByLine(img, 5)
        imgLeft = AlignImageWithH(imgLeft,hleft)
        imgRight = AlignImageWithH(imgRight,hright)
        img = CombineImages(imgLeft,imgRight)
        cv.imwrite(savePath,img)
    print('Done!')

def modifyImg():
    alignedRoot = 'resources/tmp/aligned/'
    correctedRoot = 'resources/tmp/corrected/'
    img = cv.imread('resources/tmp/aligned/1.jpg')
    ht, wd, c = img.shape
    gapLef = 28
    gapRt = 20
    gapMid = 2

    namelist = sorted(os.listdir(alignedRoot))
    for name in namelist:
        imgPath = alignedRoot+name
        savePath = correctedRoot+name
        img = cv.imread(imgPath)
        imgLF = img[0:, gapLef:int(wd/2-gapMid)]
        imgRT = img[0:, int(wd/2+gapMid):wd-gapRt]
        img=CombineImages(imgLF, imgRT)
        img = cv.resize(img, (1024, 768))
        cv.imwrite(savePath, img)
    print('DONE!')


def evaluation():
    qIm = Queue()
    qRef = Queue()
    imgRoot = 'resources/tmp/corrected/'
    refImRoot = 'resources/tmp/ref/'
    # fileNames.sort(key = lambda x:int(x[:-4]))
    refnamelist = os.listdir(refImRoot)
    refnamelist.sort(key = lambda x:int(x[:-4]))
    imgnameList = os.listdir(imgRoot)
    imgnameList.sort(key = lambda x:int(x[:-4]))

    # refImgList = []
    for i in range(9):
        for name in refnamelist:
            imgPath = refImRoot + name
            img = cv.imread(imgPath)
            # refImgList.append(img)
            qRef.put(img)

    # imgList = []
    for name in imgnameList:
        imgPath = imgRoot + name
        img = cv.imread(imgPath)
        # imgList.append(img)
        qIm.put(img)
    print('Computing evaluatin indexes...')
    evaList = []
    meanList = []
    totalssim = 0.
    totalpsnr = 0.
    totaldfid = 0.
    count = 0
    # for im, ref in zip(imgList, refImgList):
    while True:
        if not (qIm.empty() and qRef.empty()):
            count += 1
            im = qIm.get()
            ref = qRef.get()
            cv.imshow('im', im)
            cv.imshow('ref', ref)
            cv.waitKey(2000)
            cv.destroyAllWindows()
            myssim =ssim(im, ref)
            myFID = FID(im, ref)
            mypsnr = psnr(im, ref)
            evaList.append([myssim, mypsnr, myFID])
            if count!= 12:
                totalssim+=myssim
                totalpsnr+=mypsnr
                totaldfid+=myFID
            else:
                totalssim/=11
                totalpsnr/=11
                totaldfid/=11
                meanList.append([totalssim, totalpsnr, totaldfid])
                totalssim = 0.
                totalpsnr = 0.
                totaldfid = 0.
                count = 0
        else:
            break
    eva = np.array(evaList)
    mean = np.array(meanList)
    with open('test/evaluate/eva.txt', 'w') as f:
        f.write('{:<10}{:<10}{:<10}\n'.format('SSIM', 'PSNR', 'FID'))
        for line in eva:
            myssim = line[0]
            mypsnr = line[1]
            myfid = line[2]
            f.write('{:<2.4f}   {:<2.4f}    {:<2.4f}\n'.format(myssim, mypsnr, myfid))
        f.write('{:<10}{:<10}{:<10}\n'.format('MSSIM', 'MPSNR', 'MFID'))
        for line in mean:
            f.write('{:<2.4f}   {:<2.4f}    {:<2.4f}\n'.format(line[0], line[1], line[2]))
    print('DONE!')

def test():
    qIm = Queue()
    qRef = Queue()
    imgRoot = 'resources/tmp/corrected/'
    refImRoot = 'resources/tmp/ref/'

    refnamelist = os.listdir(refImRoot)
    refnamelist.sort(key=lambda x: int(x[:-4]))
    imgnameList = os.listdir(imgRoot)
    imgnameList.sort(key=lambda x: int(x[:-4]))

    # refImgList = []
    for i in range(9):
        for name in refnamelist:
            imgPath = refImRoot + name
            img = cv.imread(imgPath)
            # refImgList.append(img)
            qRef.put(img)

    # imgList = []
    for name in imgnameList:
        imgPath = imgRoot + name
        img = cv.imread(imgPath)
        # imgList.append(img)
        qIm.put(img)
    print('Computing evaluatin indexes...')
    # for im, ref in zip(imgList, refImgList):
    while True:
        if not (qIm.empty() and qRef.empty()):
            im=qIm.get()
            ref = qRef.get()
            cv.imshow('im', im)
            cv.waitKey(200)
            cv.imshow('ref', ref)
            cv.waitKey(200)
            cv.waitKey(0)
            cv.destroyAllWindows()

if __name__=='__main__':
    # tpsRoot = "resources/evaluation/ROI_Eval/TPS/"
    # rename(tpsRoot)
    tailer()
    # computeH()
    # alignAllImg()
    # modifyImg()
    # evaluation()
    # test()