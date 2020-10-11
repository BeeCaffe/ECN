from src.tools.util import *
import cv2 as cv
from queue import *
from skimage.measure import *
import time
args = {
    'img_root': 'H:\compensated goodimge\cmp3.0_res_no_gamma_dataset512\CompenNet++_l1+l2+ssim+vggLoss_4000_8_8_0.001_0.2_5000_0.0001/',
    'ref_img_toot': 'H:\compensated goodimge\desire2/',
    'res_save_root': '/',
    'img_size': (1920, 1080)
}

def evaluation(imgRoot, refImRoot,save_root ):
    qIm = Queue()
    qRef = Queue()
    refnamelist = os.listdir(refImRoot)
    imgnameList = os.listdir(imgRoot)
    for name in refnamelist:
        imgPath=refImRoot + name
        img = cv.imread(imgPath)
        img = cv.resize(img, args['img_size'])
        qRef.put(img)
    for name in imgnameList:
        imgPath = imgRoot + name
        img = cv.imread(imgPath)
        img = cv.resize(img,args['img_size'])
        qIm.put(img)
    if len(refnamelist)!=len(imgnameList):
        raise ('image and reference image number not equal')
    print('Computing evaluatin indexes...')
    evaList = []
    meanList = []
    totalssim = 0.
    totalpsnr = 0.
    totaldfid = 0.
    count = 0
    # for im, ref in zip(imgList, refImgList):
    stTm=time.time()
    while True:
        if not (qIm.empty() and qRef.empty()):
            count += 1
            im = qIm.get()
            ref = qRef.get()
            # cv2.imshow("img1",im)
            # cv2.waitKey()
            # cv2.imshow("img2",ref)
            # cv2.waitKey()
            # cv2.destroyAllWindows()
            myssim =compare_ssim(im,ref,multichannel=True)
            mymse = compare_nrmse(im, ref)
            mypsnr = compare_psnr(im, ref)
            evaList.append([myssim, mypsnr, mymse])
            if count!= len(refnamelist):
                totalssim+=myssim
                totalpsnr+=mypsnr
                totaldfid+=mymse
            else:
                totalssim/=len(refnamelist)
                totalpsnr/=len(refnamelist)
                totaldfid/=len(refnamelist)
                meanList.append([totalssim, totalpsnr, totaldfid])
                totalssim = 0.
                totalpsnr = 0.
                totaldfid = 0.
                count = 0
            nwTm=time.time()
            process("computing evaluation indexes...",count,len(refnamelist),stTm,nwTm)
        else:
            break
    eva = np.array(evaList)
    mean = np.array(meanList)
    log_name = 'ssim_psnr_rmse.txt'
    with open(log_name, 'w') as f:
        f.write('{:<10}{:<10}{:<10}\n'.format('SSIM', 'PSNR', 'RMSE'))
        for line in eva:
            myssim = line[0]
            mypsnr = line[1]
            mymse = line[2]
            f.write('{:<2.4f}   {:<2.4f}    {:<2.4f}\n'.format(myssim, mypsnr, mymse))
        f.write('{:<10}{:<10}{:<10}\n'.format('MSSIM', 'MPSNR', 'RMSE'))
        for line in mean:
            f.write('{:<2.4f}   {:<2.4f}    {:<2.4f}\n'.format(line[0], line[1], line[2]))
    print('DONE!')

if __name__=='__main__':
    evaluation(imgRoot=args['img_root'], refImRoot=args['ref_img_toot'],save_root=args['res_save_root'])