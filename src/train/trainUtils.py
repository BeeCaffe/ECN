import os
import torch.nn as nn
import torch
import torch.optim as optim
import time
import random
import PerceptualLoss as pLoss
import math
from CheckPointUtils import *
import PytorchSsim

L1Fun = nn.L1Loss()
L2Fun = nn.MSELoss()
SSIMFun = PytorchSsim.SSIM()

def optionToString(trainOption):
    """
    :brief : this method is used to transform train_option to string
    :param trainOption: train_option
    :return: string
    """
    return '{}_{}_{}_{}_{}_{}_{}_{}'.format(
        trainOption['model_name'], trainOption['train_nums'],
        trainOption['max_iters'], trainOption['batch_size'],
        trainOption['lr'], trainOption['lr_drop_ratio'],
        trainOption['lr_drop_rate'], trainOption['loss']
    )

def trainModel(net,trainData,validData,trainOption,ckpRoot):
    """
    :brief : this method is used to train and validate the model(net),and save it's trained weights to 'checkpoint'
            file.and ckpRoot is the save file root of weight and log and compensation result of CompeNet
    :param net: the CompeNet model(net), which extends to torch.nn.Model
    :param trainData: the train data which is combined by 'cam','prj','surf' three types of tensor,and each
            type with a shape of [batch_size,3,256,256]
    :param validData: the valid data
    :param trainOption: the train option list
    :param ckpRoot: the 'checkpoint' file root ,where saves the result of compensation, log file and the
            weights file (xxx.pth)
    :return net, validPSNR, validRMSE, validSSIM, trainLossBatch.item(), validLoss, weightPath:
            net ==> the CompenNet model
            validPSNR ==> the psnr of valid Dataset
            validRMSE ==> the rmse of valid Dataset
            trainLossBatch.item() ==> the train loss of train Dataset
            validLoss ==> the loss of valid Dataset
            weightPath ==> where to save the weights's file path (xxx.pth)
    """
    device = trainOption['device']
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    #train data
    trainSurf = trainData['surf']
    trainCam = trainData['cam']
    trainPrj = trainData['prj']

    #valid data
    validSurf = validData['surf']
    validCam = validData['cam']
    validPrj = validData['prj']
    params = filter(lambda param: param.requires_grad, net.parameters())
    optimizer = optim.Adam(params, lr=trainOption['lr'], weight_decay=trainOption['l2_reg'])
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=trainOption['lr_drop_rate'], gamma=trainOption['lr_drop_ratio'])
    startTime = time.time()

    #main loop
    iters = 0
    while iters < trainOption['max_iters']:
        idx = random.sample(range(trainOption['train_nums']), trainOption['batch_size'])
        trainSurfBatch = trainSurf[idx,:,:,:].to(device) if trainSurf.device.type!='cuda' else trainSurf[idx,:,:,:]
        trainCamBatch = trainCam[idx,:,:,:].to(device) if trainCam.device.type!='cuda' else trainCam[idx,:,:,:]
        trainPrjBatch = trainPrj[idx,:,:,:].to(device) if trainPrj.device.type!='cuda' else trainPrj[idx,:,:,:]

        #prediction
        net.train()
        trainCamPred = net(trainCamBatch, trainSurfBatch)
        trainLossBatch, trainL2LossBatch = computeLoss(trainCamPred, trainPrjBatch, trainOption['loss'])

        #perceptual loss
        if 'vggLoss' in trainOption['loss']:
            predictionBatch = net(trainPrjBatch, trainSurfBatch)
            perceptLossBatch =pLoss.perceptualLoss(predictionBatch, trainPrjBatch)
            trainLossBatch = trainL2LossBatch + trainOption['lambda'] * perceptLossBatch

        trainRmseBatch = math.sqrt(trainL2LossBatch.item()*3)

        # backpropagation and update params
        optimizer.zero_grad()
        trainLossBatch.backward()
        optimizer.step()

        #time
        timeLapse = time.strftime('%H:%M:%S', time.gmtime(time.time() - startTime))

        #validation
        validPSNR, validRMSE, validSSIM =0., 0., 0.
        if validData is not None and (iters%trainOption['valid_rate']==0 or iters==trainOption['max_iters']-1):
            validPSNR, validRMSE, validSSIM, validPrjPred, validLoss = evaluate(net, validData, trainOption['loss'])
        print(
            'Iter:{:5d} | Time: {} | Train Loss: {:.4f} | Train RMSE: {:.4f} | Valid PSNR: {:7s}  | Valid RMSE: {:6s}  '
            '| Valid SSIM: {:6s}  | Learn Rate: {:.5f} |'.format(iters, timeLapse, trainLossBatch.item(),
                                                                 trainRmseBatch,
                                                                 '{:>2.4f}'.format(validPSNR) if validPSNR else '',
                                                                 '{:.4f}'.format(validRMSE) if validRMSE else '',
                                                                 '{:.4f}'.format(validSSIM) if validSSIM else '',
                                                                 optimizer.param_groups[0]['lr']))
        lr_scheduler.step()
        iters+=1
    #checkpoint
    weightRoot = ckpRoot+'\\weights\\'
    if not os.path.exists(weightRoot):
        os.makedirs(weightRoot)
    title = optionToString(trainOption)
    weightPath = weightRoot + title + '.pth'
    torch.save(net.state_dict(), weightPath)
    print('Trained weight file saved to {:<s}'.format(weightPath))

    return net, validPSNR, validRMSE, validSSIM, trainLossBatch.item(), validLoss, weightPath

def computeLoss(PredBatch, PrjBatch, loss):
    """
    :brief : computing the loss of prediction and GroundTruth images, according to the content of 'loss'.
    :param PredBatch: the prediction batch , using the CompenNet
    :param PrjBatch: the GroundTruth images
    :param loss: the loss option
    :return trainLoss , L2Loss:
            trainLoss ==> it is the combine of several loss
            L2Loss ==> l2 loss
    """
    L1Loss =0
    if 'l1' in loss:
        L1Loss = L1Fun(PredBatch, PrjBatch)

    L2Loss = L2Fun(PredBatch, PrjBatch)

    SSIMLoss = 0
    if 'ssim' in loss:
        SSIMLoss = 1 * (1 - SSIMFun(PredBatch, PrjBatch))

    trainLoss = 0
    # linear combination of losses
    if loss == 'l1':
        trainLoss = L1Loss
    elif loss == 'l2':
        trainLoss = L2Loss
    elif loss == 'l1+l2':
        trainLoss = L1Loss + L2Loss
    elif loss == 'ssim':
        trainLoss = SSIMLoss
    elif loss == 'l1+ssim':
        trainLoss = L1Loss + SSIMLoss
    elif loss == 'l2+ssim':
        trainLoss = L2Loss + SSIMLoss
    elif loss == 'l1+l2+ssim':
        trainLoss = L1Loss+ L2Loss+ SSIMLoss
    elif 'vggLoss' in loss:
        pass
    else:
        print('Unsupported loss')

    return trainLoss, L2Loss

def evaluate(net, validData, loss):
    """
    :brief : this function is used to evaluate the pictures's qualities of validation dataset, the evaluation index of qualities
            are 'psnr', 'rmse', 'ssim' ,'loss'
    :param net:
    :param validData:
    :param loss:
    :return:
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    validSurf = validData['surf']
    validCam = validData['cam']
    validPrj = validData['prj']

    with torch.no_grad():
        net.eval()
        validLoss = 0.
        if validSurf.device.type != device.type:
            lastLoc = 0
            validMSE, validSSIM = 0., 0.
            pred = torch.zeros(validPrj.shape)
            validNum = validCam.shape[0]
            batchSize = 50 if validNum > 50 else validNum  # default number of valid images per dataset

            for i in range(0, validNum // batchSize):
                idx = range(lastLoc, lastLoc + batchSize)
                validSurfBatch = validSurf[idx, :, :, :].to(device) if validSurf.device.type != 'cuda' else validSurf[idx, :, :, :]
                validCamBatch = validCam[idx, :, :, :].to(device) if validCam.device.type != 'cuda' else validCam[idx, :, :, :]
                validPrjBatch = validPrj[idx, :, :, :].to(device) if validPrj.device.type != 'cuda' else validPrj[idx, :, :, :]

                # predict batch
                predBatch = net(validCamBatch, validSurfBatch)

                pred[lastLoc:lastLoc + batchSize, :, :, :] = predBatch.cpu()

                # compute loss
                validMSE += L2Fun(predBatch, validPrjBatch).item() * batchSize
                validSSIM += ssim(predBatch, validPrjBatch) * batchSize
                validLoss, L2Loss= computeLoss(predBatch, validPrjBatch, loss)
                validLoss += validLoss
                lastLoc += batchSize
            # average
            validLoss /= validNum
            validMSE /= validNum
            validSSIM /= validNum
            validRMSE = math.sqrt(validMSE * 3)
            validPSNR = 10 * math.log10(1 / validMSE)
        else:
            pred = net(validCam, validSurf)
            validMSE = L2Fun(pred, validPrj).item()
            validRMSE = math.sqrt(validMSE * 3)
            validPSNR = 10 * math.log10(1 / validMSE)
            validSSIM = SSIMFun(pred, validPrj).item()
            validLoss, L2Loss = computeLoss(pred, validPrj, loss)

    return validPSNR, validRMSE, validSSIM, pred, validLoss

def ssim(x, y):
    if x.shape[0] != y.shape[0]:
        raise RuntimeError('the batch size of x and y is not equal!')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num = x.shape[0]
    batch_size = 200
    lastLoc = 0
    SSIM = 0.
    iters = num // batch_size
    if num <= batch_size:
        # device = torch.device('cpu')
        x = x.to(device) if x.device.type != device.type else x
        y = y.to(device) if y.device.type != device.type else y
        with torch.no_grad():
            l2_fun = nn.MSELoss()
            SSIM = PytorchSsim.ssim(x, y).item()
    else:
        for i in range(iters):
            newx = x[lastLoc:lastLoc + batch_size, :, :, :].to(device) if x.device.type != device.type else x[
                                                                                                            lastLoc:lastLoc + batch_size,
                                                                                                            :, :, :]
            newy = y[lastLoc:lastLoc + batch_size, :, :, :].to(device) if y.device.type != device.type else y[
                                                                                                            lastLoc:lastLoc + batch_size,
                                                                                                            :, :, :]
            with torch.no_grad():
                l2_fun = nn.MSELoss()
                ssimBatch = PytorchSsim.ssim(newx, newy).item()
            lastLoc += batch_size
            SSIM += ssimBatch / iters
    return SSIM

def psnr(x, y):
    if x.shape[0] != y.shape[0]:
        raise RuntimeError('the batch size of x and y is not equal!')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num = x.shape[0]
    batch_size = 200
    lastLoc = 0
    PSNR = 0.
    iters = num//batch_size
    if num<=batch_size:
        # device = torch.device('cpu')
        x = x.to(device) if x.device.type != device.type else x
        y = y.to(device) if y.device.type != device.type else y
        with torch.no_grad():
            l2_fun = nn.MSELoss()
            PSNR = 10 * math.log10(1 / l2_fun(x, y))
    else:
        for i in range(iters):
            newx = x[lastLoc:lastLoc+batch_size,:,:,:].to(device) if x.device.type != device.type else x[lastLoc:lastLoc+batch_size,:,:,:]
            newy = y[lastLoc:lastLoc+batch_size,:,:,:].to(device) if y.device.type != device.type else y[lastLoc:lastLoc+batch_size,:,:,:]
            with torch.no_grad():
                l2_fun = nn.MSELoss()
                psnrBatch = 10 * math.log10(1/l2_fun(newx, newy))
            lastLoc += batch_size
            PSNR += psnrBatch/iters
    return PSNR

def rmse(x, y):
    if x.shape[0] != y.shape[0]:
        raise RuntimeError('the batch size of x and y is not equal!')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num = x.shape[0]
    batch_size = 200
    lastLoc = 0
    RMSE = 0.
    iters = num//batch_size
    if num<=batch_size:
        # device = torch.device('cpu')
        x = x.to(device) if x.device.type != device.type else x
        y = y.to(device) if y.device.type != device.type else y
        with torch.no_grad():
            l2_fun = nn.MSELoss()
            RMSE = math.sqrt(l2_fun(x, y).item() * 3)
    else:
        for i in range(iters):
            newx = x[lastLoc:lastLoc+batch_size,:,:,:].to(device) if x.device.type != device.type else x[lastLoc:lastLoc+batch_size,:,:,:]
            newy = y[lastLoc:lastLoc+batch_size,:,:,:].to(device) if y.device.type != device.type else y[lastLoc:lastLoc+batch_size,:,:,:]
            with torch.no_grad():
                l2_fun = nn.MSELoss()
                rmseBatch = math.sqrt(l2_fun(newx, newy).item() * 3)
            lastLoc += batch_size
            RMSE += rmseBatch/iters
    return RMSE

