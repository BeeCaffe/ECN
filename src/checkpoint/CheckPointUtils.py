from time import localtime,strftime
import os
def newLogFile(logFileRoot, train_option):
    """
    :brief : passing the log file's root path and train option list, this function will produce a log file in
            'logFileRoot' and return the log file path.
    :param logFileRoot: the root directory of log file ,where you suppose to save your log file, if the file
            path not exist, it will bulid a file for it.
    :param train_option: the train options list
    :return: log file's path ,you can write log in this path
    """
    if not os.path.exists(logFileRoot):
        os.makedirs(logFileRoot)
    nameStr = '_{}_{}_{}_{}'.format(train_option['train_nums'],
                                                 train_option['max_iters'],
                                                 train_option['batch_size'],
                                                 train_option['lr']
                                                 )
    timeStr = strftime('%Y-%m-%d %H-%M-%S', localtime())
    logFileName = timeStr+nameStr+'.txt'
    logFile = open(logFileRoot+logFileName, 'w')
    titleStr = '{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}\n'
    logFile.write(titleStr.format('ModelName', 'LossFunction',
                                  'NumTrain', 'BatchSize', 'MaxIters',
                                  'Lambda',
                                  'UncmpPSNR', 'UncmpRMSE', 'UncmpSSIM',
                                  'ValidPSNR', 'ValidRMSE', 'ValidSSIM',
                                  'TrainLoss', 'ValidLoss'))
    logFile.close()
    return logFileRoot+logFileName