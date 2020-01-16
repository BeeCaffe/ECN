import os
import torch
import cv2 as cv
import numpy as np
import ECN
MaxNum = 2000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args={
    "rows":1080,
    'cols':1920
}

paths = {
    'pth_root': 'I:\Backup\weights\gamma1.5/',#the model or weights save file root path of CompeNet
    'pth_name': 'CompeNet_5000_2000_4_0.001_0.2_800_l1+l2+ssim+vggLoss.pth',

    'surf_path': 'F:\yandl\CompenNet-plusplus3.0\data\dataset_512_gama2.2\cmp\surf\surf.jpg',#the surface image path

    'prj_data_root': 'F:\yandl\CompenNet-plusplus3.0\data\dataset_512_gama2.2\cmp\desire\\',#the GroundTruth images file root path

    'save_root': 'H:\compensated goodimge\ECN\\',#where you suppose to save you compensated images
}

namelists=os.listdir(paths.get('prj_data_root'))

pth_path = paths['pth_root']+paths['pth_name']

#get the data root of GroundTruth's pictures

surf_img = cv.cvtColor(cv.resize(cv.imread(paths['surf_path']),(args.get('cols'), args.get('rows'))), cv.COLOR_BGR2RGB)
surf_img = torch.from_numpy(surf_img).expand([1, args.get('rows'), args.get('cols'), 3])
surf_img = surf_img.permute((0, 3, 1, 2)).float().div(255)


save_name = paths['pth_name'][:-4]+'/'
save_path = paths['save_root']+save_name

#read each pictures in prjector data root
name_list = os.listdir(paths['prj_data_root'])
name_list.sort(key=lambda x: int(x[:-4]))
for i, name in enumerate(name_list):
    #read each picture and feed it to CompeNet Model
    prj_img = cv.cvtColor(cv.resize(cv.imread(paths['prj_data_root']+name), (args.get('cols'), args.get('rows'))), cv.COLOR_BGR2RGB)
    prj_img = torch.from_numpy(prj_img).expand([1, args.get('rows'), args.get('cols'), 3])
    prj_img = prj_img.permute((0, 3, 1, 2)).float().div(255)
    #load model
    model = ECN.ECN().cuda()
    model.load_state_dict(torch.load(pth_path))
    pred = model(prj_img.to(device), surf_img.to(device))
    #save pred_l1_125_5000 to you desired path
    pred = pred[0, :, :, :]
    pred = np.uint8((pred[:, :, :] * 255).permute(1, 2, 0).cpu().detach().numpy())
    pred = pred[:, :, ::-1]

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    cv.imwrite(save_path+str(i)+'.jpg', pred)
    print("has componeted {:<d} pictures".format(i))
    if i > MaxNum:
        exit(0)

