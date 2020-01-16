from trainUtils import *
import DatasetUtils as utils
import torch
import ECN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Valid with', torch.cuda.device_count(), 'GPUs!') if torch.cuda.device_count() >= 1 else print('Train with CPUs!')

pth_path = 'checkpoint/weights/CompeNet_5000_3000_2_0.001_0.2_800_l1+l2+ssim+vggLoss.pth'
dataset_name = 'Tps/'
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

#train option
valid_option = {'model_name': 'ECN',
                'train_nums': 5000,  #How many images you want to train
                'max_iters': 3000,
                'batch_size': 2,
                'loss': 'l1+l2+ssim+vggLoss',
                'lr': 1e-3,  # learning rate
                'lr_drop_ratio': 0.2,
                'lr_drop_rate': 800,
                'loss': '',  # loss will be set to one of the loss functions in loss_list later
                'l2_reg': 1e-4,  # l2 regularization
                'device': device,
                'train_plot_rate': 100,  # training and plot rate
                'valid_rate': 100,  # validation
                'save_compensation': True,
                'lambda': 1}
logFilePath = 'checkpoint/evalLog/'+dataset_name[:-1]+'.txt'
with open(logFilePath, 'w') as logFile:
    titleStr = '{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}\n'
    logFile.write(titleStr.format('ModelName', 'LossFunction',
                              'NumTrain', 'BatchSize', 'MaxIters',
                              'Lambda',
                              'ValidPSNR', 'ValidRMSE', 'ValidSSIM'))

validData = utils.loadValidDataset(dataset_root, paths)

startTime = time.time()

timeLapse = time.strftime('%H:%M:%S', time.gmtime(time.time() - startTime))

net = ECN.ECN().cuda()
net.load_state_dict(torch.load(pth_path))

# validation
validPSNR, validRMSE, validSSIM = 0., 0., 0.
if validData is not None :
    validPSNR, validRMSE, validSSIM, validPrjPred, validLoss = evaluate(net, validData, valid_option['loss'])
print('Iter:{:5d} | Time: {}| Valid PSNR: {:7s}  | Valid RMSE: {:6s}  '
      '| Valid SSIM: {:6s}  |'.format(1, timeLapse, '{:>2.4f}'.format(validPSNR) if validPSNR else '',
                                                 '{:.4f}'.format(validRMSE) if validRMSE else '',
                                                 '{:.4f}'.format(validSSIM) if validSSIM else ''))
with open(logFilePath, 'a') as logFile:
    titleStr = '{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}{:<2.4}{:<2.4}{:<2.4}\n'
    logFile.write(titleStr.format(valid_option['model_name'], valid_option['loss'],
                                  valid_option['train_nums'], valid_option['batch_size'],
                                  valid_option['max_iters'], valid_option['lambda'],
                                  validPSNR, validRMSE,
                                  validSSIM, ))
