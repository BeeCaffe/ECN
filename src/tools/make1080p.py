import os
import cv2
import util
import time

def make1080p(fileRoot,saveRoot):
    #如果保存路径不存在，则创建一个
    if not os.path.exists(saveRoot):
        os.mkdir(saveRoot)
    nameLists=os.listdir(fileRoot)
    nameLists.sort(key = lambda x:int(x[:-4]))
    total=len(nameLists)
    number=0
    startTime=time.time()
    #筛选图片，只要长度大于宽度的图片
    for i,imgName in enumerate(nameLists):
        imgPath=fileRoot+imgName
        img=cv2.imread(imgPath, cv2.IMREAD_UNCHANGED)
        #如果图片为空，跳过
        if img is None:
            continue
        cols, rows, c=img.shape
        if rows/cols < 1:
            continue
        else:
            img=cv2.resize(img,(args.get('COLS'),args.get('ROWS')))
            savePath=saveRoot+str(number)+'.jpg'
            cv2.imwrite(savePath, img)
            nowTime=time.time()
            number+=1
            util.process('dealing images ', i, total, startTime, nowTime)

#make 1080p images
if __name__=='__main__':
    args = {
        'COLS': 1920,
        'ROWS': 1080,
        'imgRoot': 'E:\Backup\Images\HDRImages\\',
        'saveRoot': 'E:\Backup\Images\images1920x1080\\'
    }
    make1080p(args.get('imgRoot'), args.get('saveRoot'))
    print('DONE!')

