class DetectedItem :
    def __init__(self, cordinates, classifcation, confidence, image, startX, startY) -> None:
        imH, imW, imC = image.shape
        self.xmin = startX + int(cordinates[1] * imW)
        self.ymin = startY + int(cordinates[0] * imH)
        self.xmax = startX + int(cordinates[3] * imW)
        self.ymax = startY + int(cordinates[2] * imH)
        self.classifcation = classifcation
        self.confidence = confidence
