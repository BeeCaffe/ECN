import cv2 as cv
import os
import numpy as np

# data_root = "H:\\python\\MyCompenNet\\Cam_One_Two_Data\\OneScreen"
# save_root = "H:\\python\\MyCompenNet\\Cam_One_Two_Data\\OneScreen256"

# data_root = "H:\\python\\MyCompenNet\\Cam_One_Two_Data\\TwoScreen"
# save_root = "H:\\python\\MyCompenNet\\Cam_One_Two_Data\\TwoScreen256"

data_root = "H:\\python\\MyCompenNet\\GammaCorData\\OneCorImages"
save_root = "H:\\python\\MyCompenNet\\GammaCorData\\OneCorImages256"
"""
@brief this program is used to split the 1024x1024 images into several 256x256 images ,and rename them follow the order.
        modify data_root and save_root you can process the images you want to.
@global data_root-the file root of the images , where you want to split.
        save_root-the file path of you want to save splitted images.
"""
def split(data_root, save_root, maxNums):

    name_list = os.listdir(data_root)

    name_list.sort(key=lambda x:int(x[:-4]))

    count = 0

    if not os.path.exists(save_root):

        os.mkdir(save_root)

    for name in name_list:
        if count <= maxNums:
            img_path = data_root+os.sep+name

            img = cv.imread(img_path,cv.IMREAD_UNCHANGED)

            img = cv.resize(img,(768,1024))

            imgs = splitto256(img)

            for img in imgs:

                count+=1

                save_path = save_root+os.sep+str(count)+".jpg"


                cv.imwrite(save_path,img)

                print("Having Processed {:<d} Images".format(count))

def splitto256(img):

    img_list = []

    height = img.shape[0]

    width = img.shape[1]

    img_np = np.array(img,dtype=np.uint)

    h = int(height/256)
    w = int(width/256)

    for i in range(h):
        for j in range(w):

            img_list.append(img[i*256:(i+1)*256, j*256:(j+1)*256])

    return img_list

if __name__=="__main__":
    split(data_root, save_root)
