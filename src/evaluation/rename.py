import os
import cv2 as cv
args = {
    'src_root':'H:\compensated goodimge\compenNet\CompeNet_5000_3000_2_0.001_0.2_800_l1+ssim/',
    'save_root':'H:\compensated goodimge\compenNet2\CompeNet_5000_3000_2_0.001_0.2_800_l1+ssim/'
}

def rename(src_root,save_root):
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    namelists=sorted(os.listdir(src_root))
    count=1
    for name in namelists:
        img_path=src_root+name
        img=cv.imread(img_path)
        nwname = 'img_'
        if count<10:
            nwname+='000'+str(count)+'.jpg'
        elif count>=10:
            nwname+='00'+str(count)+'.jpg'
        cv.imwrite(save_root+nwname,img)
        count+=1
    print("Done!")

if __name__=='__main__':
    rename(src_root=args['src_root'],save_root=args['save_root'])