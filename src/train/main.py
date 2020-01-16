import DatasetUtils as utils
import CheckPointUtils as ckp
import ECN as net
import CompesateUtils as comp
from trainUtils import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Train with', torch.cuda.device_count(), 'GPUs!') if torch.cuda.device_count() >= 1 else print('Train with CPUs!')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
dataset_name = 'Ours'
dataset_root = 'resources/eavaluateDataset/'+dataset_name
paths = [
    '/train/cam/',#train images
    '/train/prj/',
    '/train/surf/',

    '/valid/cam/',#valid images
    '/valid/prj/',
    '/valid/surf/',

    '/cmp/prj/',#compensation images
    '/cmp/surf/',

    '/res/prj'#compensated images
]
#params list
loss_list = ['l1', 'l1+l2', 'l1+ssim', 'l2+ssim', 'l1+l2+ssim', 'l1+vggLoss', 'l2+vggLoss',
             'l1+l2+vggLoss', 'l1+ssim+vggLoss', 'l2+ssim+vggLoss', 'l1+l2+ssim+vggLoss']

# loss_list=['l1+vggLoss', 'l2+vggLoss',
#              'l1+l2+vggLoss', 'l1+ssim+vggLoss', 'l2+ssim+vggLoss', 'l1+l2+ssim+vggLoss']

model_list = ['ECN']

#train option
train_option = {'model_name': '',
                'train_nums': 5000,     #How many images you want to train
                'max_iters': 3000,
                'batch_size': 2,
                'lr': 1e-3,  # learning rate
                'lr_drop_ratio': 0.2,
                'lr_drop_rate': 800,
                'loss': '',     # loss will be set to one of the loss functions in loss_list later
                'l2_reg': 1e-4,     # l2 regularization
                'device': device,
                'train_plot_rate': 100,     # training and plot rate
                'valid_rate': 100,      # validation
                'save_compensation': True,
                'lambda': 1}

#is compesate?
isCompensate = True

#loading data
print("Loading dataset from {:<}".format(dataset_root))
trainData, validData = utils.LoadTrainAndValid(dataset_root, paths, train_option['train_nums'])

#log file
ckp_root = os.path.join(os.getcwd(), '..'+os.sep+'..'+os.sep+'checkpoint')
logFileRoot = ckp_root + '\\log\\'
logFilePath = ckp.newLogFile(logFileRoot, train_option)
print('Log file builds in {:<}'.format(logFilePath))

#main
for modelName in model_list:
    train_option['model_name'] = modelName
    for loss in loss_list:
        logFile = open(logFilePath, 'a')
        model = net.ECN()
        model.to(device)
        train_option['loss'] = loss
        print('-------------------------------------- Training Options -----------------------------------')
        print("\n".join("{}: {}".format(k, v) for k, v in train_option.items()))
        print('-------------------------------------- Start training CompenNet ---------------------------')
        try:
            model, validPSNR, validRMSE, validSSIM, trainLoss, validLoss, pthPath = trainModel(model,
                                                                                      trainData,
                                                                                      validData,
                                                                                      train_option,
                                                                                      ckp_root)
        except RuntimeError as e:
            if 'out of memory ' in str(2):
                print('| WARNING: ran out of memory')
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else: raise e
        #save result to log file
        resStr = '{:<20}{:<20}{:<20}{:<20}{:<20}{:<20.4f}{:<20.4f}{:<20.4f}{:<20.4f}{:<20.4f}{:<20.4f}{:<20.4f}{:<20.4f}\n'
        logFile.write(resStr.format(modelName, loss,
                                    train_option['train_nums'], train_option['batch_size'],
                                    train_option['max_iters'],train_option['lambda'], psnr(validData['cam'], validData['prj']),
                                    rmse(validData['cam'], validData['prj']), ssim(validData['cam'], validData['prj']),
                                    validPSNR, validRMSE,
                                    validSSIM, trainLoss,
                                    validLoss
                                    ))
        logFile.close()
        #compensate images
        if isCompensate:
            print('Saving compensated images ...\n')
            torch.cuda.empty_cache()
            with torch.no_grad():
                torch.cuda.empty_cache()
                prjRoot = dataset_root+paths[6]
                # surfPath = dataset_root+paths[7]+'surf.jpg'
                surfRoot = dataset_root+paths[7]
                pthPath = pthPath
                saveName = train_option['loss']
                saveRoot = ckp_root+'\\res\\'+saveName+'\\'
                maxNum = 10
                nameList = os.listdir(surfRoot)
                for name in nameList:
                    surfPath = surfRoot+name
                    comp.compensate(prjRoot, surfPath, pthPath, saveRoot+name[:-4]+'\\', maxNum)
        del model
        torch.cuda.empty_cache()
        print('Compensated Pictures Have Been Saved To {:<s}'.format(saveRoot))
    del trainData
    del validData
    print('--------------------------------All Things Done! ---------------------------\n')