import os
class BrightChannelBean():
    def __init__(self, imgPath):
        self.imgPath = imgPath
        self.saveDir = "res/BrightChannelImages/Generate/"
        if not os.path.exists(self.saveDir):
            os.makedirs(self.saveDir)