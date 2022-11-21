import cv2, time, os, tensorflow as tf
from os.path import exists
import numpy as np

from tensorflow.python.keras.utils.data_utils import get_file

class Detector:
    def __init__(self) -> None:
        pass
    def readClasses(self,classesFilePath):
        with open(classesFilePath, 'r') as f:
            self.classesList = f.read().splitlines()
        
        self.colorList = np.random.uniform(low = 0, high= 255, size=(len(self.classesList), 3))

        print (len(self.classesList), len(self.colorList))
    
    def downLoadModel(self, modelURL) :
        fileName = os.path.basename(modelURL)
        self.modelName = fileName[:fileName.index('.')]
        self.cacheDir = "./pretrainedModels"

        # self.modelName = 
        
        if (os.path.exists("pretrainedModels\checkpoints\\" + fileName)) :
            return

        os.makedirs(self.cacheDir, exist_ok=True)

        get_file(fname=fileName,
        origin=modelURL, cache_dir=self.cacheDir, cache_subdir="checkpoints",extract= True)
    
    def loadModel(self):
        print ("loading model " + self.modelName)

        # Triggers error
        # tf.keras.backend.clear_session()

        self.model = tf.saved_model.load(os.path.join(self.cacheDir, "checkpoints", self.modelName, "saved_model"))

        print ("Model " + self.modelName + " loaded successfully")
    
    def createBoundigBoz(self,image, threshold = 0.5):
        inputTensor = cv2.cvtColor(image.copy(),cv2.COLOR_BGR2RGB)
        inputTensor = tf.convert_to_tensor(inputTensor, dtype=tf.uint8)
        inputTensor = inputTensor[tf.newaxis,...]
        # inputTensor = inputTensor[:, :, :] # <= add this line


        detections = self.model(inputTensor)

        bboxs = detections['detection_boxes'][0].numpy()
        classIndexes = detections['detection_classes'][0].numpy().astype(np.int32)
        classScores = detections['detection_scores'][0].numpy()

        imH, imW, imC = image.shape

        bboxIdx = tf.image.non_max_suppression(bboxs,classScores,max_output_size=50,
        iou_threshold=threshold, score_threshold=threshold)

        print ("bbox id : "+str(bboxIdx))

        if len(bboxIdx) != 0 :
            for i in bboxIdx :
                bbox = tuple(bboxs[i].tolist())
                classConfidence = round(100*classScores[i])
                classIndex = classIndexes[i] -1

                classLabelText = self.classesList[classIndex]

                classColor = self.colorList[classIndex]

                if (classLabelText == "person") : 
                    classLabelText = "hoe"
                    print("whore")

                displayText = '{}: {}%'.format(classLabelText,classConfidence)

                ymin, xmin, ymax, xmax = bbox

                xmin, xmax, ymin, ymax = (xmin * imW, xmax *imW, ymin *imH, ymax * imH)

                xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)

                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color = classColor, thickness= 1 )
                cv2.putText(image, displayText, (xmin,ymin - 10), cv2.FONT_HERSHEY_PLAIN, 1 , classColor, 2)
            return image

                # print (ymin, xmin, ymax, xmax)
                # break


    def predictImage(self, imagePath, threshold) :
        image = cv2.imread(imagePath)
        # cv2.imshow("Result", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()



        bboxImage = self.createBoundigBoz(image, threshold)

        cv2.imwrite(self.modelName + ".jpg", bboxImage)

        cv2.imshow("Result", bboxImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
