class ImageBean:
    def __init__(self,
                 srcPath,
                 ourPath,
                 unCorrectedPath,
                 CompenNetPath,
                 TpsPath,
                 BimberPath,
                 saveDir):
        self.srcPath = srcPath
        self.ourPath = ourPath
        self.unCorrectedPath = unCorrectedPath
        self.CompenNetPath = CompenNetPath
        self.TpsPath = TpsPath
        self.BimberPath = BimberPath
        self.saveDir = saveDir