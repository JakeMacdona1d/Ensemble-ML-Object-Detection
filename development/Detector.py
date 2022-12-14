import cv2, time, os, tensorflow as tf
from os.path import exists
import pathlib
import numpy as np

from development.SubClassDetector import *
from development.vidOutput import *

from tensorflow.python.keras.utils.data_utils import get_file

class Detector:
    # -> None ensures that the constructor returns None value
    def __init__(self) -> None:
        subDetectors = [] 

    def readClasses(self,classesFilePath):
        with open(classesFilePath, 'r') as f:
            self.classesList = f.read().splitlines()     
        self.colorList = np.random.uniform(low = 0, high= 255, size=(len(self.classesList), 3))

    def downLoadModel(self, modelURL) :
        fileName = os.path.basename(modelURL)
        self.modelName = fileName[:fileName.index('.')]
        self.cacheDir = "./development/models/exported-models/pretrainedModels"
        if (os.path.exists("development/models/exported-models/pretrainedModels/checkpoints/" + fileName)) :
            return

        os.makedirs(self.cacheDir, exist_ok=True)

        get_file(fname=fileName,
        origin=modelURL, cache_dir=self.cacheDir, cache_subdir="checkpoints",extract= True)
    
    def loadModel(self):
        tf.keras.backend.clear_session()
        print('Loading model...', end='')
        start_time = time.time()
        self.model = tf.saved_model.load(os.path.join(self.cacheDir, "checkpoints", self.modelName, "saved_model"))
        end_time = time.time()
        elapsed_time = end_time - start_time
        print('Done! Took {} seconds'.format(elapsed_time))
    
    # This bounding box function differs from the other beacause it was the downloaded 
    #and model lacks certain features that are available in our custum training.
    def createBoundigBoz(self, imagePath, threshold = 0.5):
        image = None
        video = None
        images = []
        if pathlib.Path(imagePath).suffix == ".mp4" :
            video = cv2.VideoCapture(imagePath)
        else : 
            image = cv2.imread(imagePath)

        self.humanModel = SubClassDetector()

        self.humanModel.loadModel("development/models/exported-models/my_mobilenet_model/saved_model")
        self.humanModel.setPathToModel("models/exported-models/my_mobilenet_model/saved_model/label_map.pbtxt") 

        playing = True
        while playing :
            if not video == None :
                 ret, image = video.read()
                 if not ret : break
            else : playing = False

            inputTensor = cv2.cvtColor(image.copy(),cv2.COLOR_BGR2RGB)
            inputTensor = tf.convert_to_tensor(inputTensor, dtype=tf.uint8)
            inputTensor = inputTensor[tf.newaxis,...]

            detections = self.model(inputTensor)

            bboxs = detections['detection_boxes'][0].numpy()
            classIndexes = detections['detection_classes'][0].numpy().astype(np.int64)
            classScores = detections['detection_scores'][0].numpy()

            imH, imW, imC = image.shape

            bboxIdx = tf.image.non_max_suppression(bboxs,classScores,max_output_size=50,
            iou_threshold=threshold, score_threshold=threshold)

            if len(bboxIdx) != 0 :
                for i in bboxIdx :
                    bbox = tuple(bboxs[i].tolist())
                    classConfidence = round(100*classScores[i])
                    classIndex = classIndexes[i] -1

                    classLabelText = self.classesList[classIndex]
                    classColor = self.colorList[classIndex]
                    displayText = '{}: {}%'.format(classLabelText,classConfidence)
                    ymin, xmin, ymax, xmax = bbox
                    xmin, xmax, ymin, ymax = (int(xmin * imW), int(xmax *imW), int(ymin *imH), int(ymax * imH))

                    if (classLabelText == "person") : 
                        # Slicing to crop the image
                        crop = image[ymin:int(ymax),xmin:xmax ]
                        self.humanModel.specify(crop, xmin, ymin)
                        displaySpec = '{}: {}%'.format(self.humanModel.item.classifcation, self.humanModel.item.confidence)

                        if (self.humanModel.item.confidence > threshold) :
                            cv2.rectangle(image,(self.humanModel.item.xmin, self.humanModel.item.ymin),
                             (self.humanModel.item.xmax, self.humanModel.item.ymax),
                              color = self.colorList[classIndex + 1],thickness= 2) 

                            cv2.putText(image, displaySpec, (self.humanModel.item.xmin, self.humanModel.item.ymin - 10),
                             cv2.FONT_HERSHEY_PLAIN, 1 , self.colorList[classIndex + 1], 2)
                        
                    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color = classColor, thickness= 2 )
                    cv2.putText(image, displayText, (xmin,ymin - 10), cv2.FONT_HERSHEY_PLAIN, 1 , classColor, 2)
            images.append(image)
        if playing : return images
        return image

    
    def predictImage(self, mediaPath, threshold) :
        rawMedia = self.createBoundigBoz(mediaPath, threshold)
        makeMedia(rawMedia)

