from src.tools.util import *
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

title = 'image\'s ROI analysis'
# title = 'Different Loss Functions'
# title = 'Gamma Correction'

# title = 'Different Lambda'

Rows = 768
Cols = 1024

Fontdict={'family': 'Times New Roman',
          'color': 'black',
          'weight': 'normal',
          'size': 14}
RegionSize = [300, 300]
Points = [(100, 100), (100, 450),
          (600, 100), (600, 450)]
Color = ['green', 'yellowgreen', 'red', 'gold', 'lightskyblue', 'lightcoral']
Step = 20
YLim = (-10, 250)
XLim = (0, 80)
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
Labels = ['Ground truth', 'UnCompensated', 'Ours', 'CompenNet', 'Tps', 'O.Bimber\'s']
# Labels = ['Ground truth','UnCompensated','ours','l1+ssim+lpre','l1+l2+ssim' ,'l1+ssim']
# Labels = ['Ground truth', 'UnCompensated', 'Gamma Corrected', 'Without Gamma Correction']
# Labels = ['Ground truth', 'UnCompensated', 'our lambda=0.1', 'lambda=1','lambda=0.01','lambda=0.001']

def drawROI(imgList,channel):
    if channel!=0:
        pltname = 'ROI_'+str(channel)+'.png'
    elif channel==0:
        pltname = 'ROI.png'
    for i in range(4):
        cv.rectangle(imgList[1], pt1=Points[i],pt2=(Points[i][0]+RegionSize[0], Points[i][1]+RegionSize[0]), color=(255,255,0),thickness=2)
    cv.imshow("img", imgList[1])
    cv.waitKey(0)
    cv.destroyAllWindows()
    fig = plt.figure(num='FOI-analysis', figsize=(8, 5), dpi=75, facecolor='#FFFFFF', edgecolor='#0000FF')
    plt.title(title)
    ax = plt.gca()
    ax.set_ylim(YLim)
    plt.xlim(XLim)
    plt.xticks(np.linspace(XLim[0], XLim[1], 1, endpoint=True))
    ax.set_yticks(np.linspace(YLim[0], YLim[1], 10, endpoint=True))
    plt.ylabel('Value', fontdict=Fontdict)
    yMajorLocator = MultipleLocator(20)
    ax.yaxis.set_major_locator(yMajorLocator)

    for i, img in enumerate(imgList):
        x = 2*i+5
        if i < len(Points) : plt.text(x+1+i*(Step-1), YLim[0]-10, 'region_'+str(i+1), fontdict=Fontdict)
        plt.vlines(0, 0, 0, colors=Color[i], label=Labels[i])
        for pt in Points:
            rg = GetRegion(img, pt=pt, width=RegionSize[0], height=RegionSize[1])
            roi=0
            if channel!=0:
                roi = ROI(rg[:, :, channel-1],0)
            elif channel==0:
                roi = ROI(rg, 1)
            plt.plot(x, roi[0], 'o', c=Color[i])
            plt.plot(x, roi[0] - roi[1] / 2, '_', c=Color[i])
            plt.plot(x, roi[0] + roi[1] / 2, '_', c=Color[i])
            plt.vlines(x, roi[0] - roi[1] / 2, roi[0] + roi[1] / 2, colors=Color[i])
            x += Step
    plt.legend()
    plt.savefig(pltname)
    plt.close(fig)

# evaluate different traditional method
if __name__=='__main__':
    id=17
    grudTuth = cv.resize(cv.imread('resources/eval3/correct/projector/'+str(id)+'.jpg'), (Cols, Rows))
    uncompen = cv.resize(cv.imread('resources/eval3/correct/Groundtruth/'+str(id)+'.JPG'), (Cols, Rows))
    compenNet = cv.resize(cv.imread('resources/eval3/correct/CompenNet/'+str(id)+'.JPG'), (Cols, Rows))
    ours = cv.resize(cv.imread('resources/eval3/correct/Ours/'+str(id)+'.JPG'), (Cols, Rows))
    tps = cv.resize(cv.imread('resources/eval3/correct/Tps/'+str(id)+'.JPG'), (Cols, Rows))
    scatter = cv.resize(cv.imread('resources/eval3/correct/Scatter/'+str(id)+'.JPG'), (Cols, Rows))
    imgList = [grudTuth, uncompen, ours, compenNet, tps, scatter]
    for i in range(4):
        drawROI(imgList, i)

# #evaluate different loss
# if __name__=='__main__':
#     groundTruth = cv2.imread('resources/eval/loss/4.jpg')
#
#     uncompen = cv2.imread('resources/eval/loss/uncompen.jpg')
#     loss1 = cv2.imread('resources/eval/loss/l1+ssim.jpg')
#     loss2 = cv2.imread('resources/eval/loss/l1+ssim+vgg.jpg')
#     loss3 = cv2.imread('resources/eval/loss/l1+l2+ssim.jpg')
#     loss4 = cv2.imread('resources/eval/loss/ours.jpg')
#     imList =[groundTruth,uncompen,loss4,loss2,loss3,loss1]
#     for i in range(4):
#         drawROI(imList, i)

# #evaluate different lambda
# if __name__=='__main__':
#     groundTruth = cv2.imread('resources/eval/lambda/8.jpg')
#     uncompen = cv2.imread('resources/eval/lambda/uncompen.jpg')
#     lambda_1 = cv2.imread('resources/eval/lambda/1e1.jpg')
#     lambda_2 = cv2.imread('resources/eval/lambda/1e-1.jpg')
#     lambda_3 = cv2.imread('resources/eval/lambda/1e-2.jpg')
#     lambda_4 = cv2.imread('resources/eval/lambda/1e-3.jpg')
#     imList =[groundTruth,uncompen,lambda_2,lambda_1,lambda_3,lambda_4]
#     for i in range(4):
#         drawROI(imList, i)

# #evaluate gamma
# if __name__=='__main__':
#     groundTruth = cv2.imread('resources/eval/gamma/6.jpg')
#     uncompen = cv2.imread('resources/eval/gamma/uncompen.jpg')
#     gamma = cv2.imread('resources/eval3/correct/Ours/6.JPG')
#     withoutGamma = cv2.imread('resources/eval/gamma/withoutGamma.jpg')
#     imList =[groundTruth,uncompen,gamma,withoutGamma]
#     for i in range(4):
#         drawROI(imList, i)
