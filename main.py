# https://www.youtube.com/watch?v=2yQqg_mXuPQ
from development.Detector import *

# https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbGRRLXF6ZlhXcnVsbDZzX0ZiOWprN2tMYUNBUXxBQ3Jtc0tsNTRBQnBtaHFOVE9iYjhCcFVaOTNvOG1hUUhLSlhUaS1mcWZzZERNZ2VyOHkyYzdXbzhyN2FqM283TnRPclpEellQZ1pxb3pndG84bWpQNlA3bEUyRmxRb3dDbk5lOWh4VjRiQVVSZzBLZ1JZbm1IYw&q=https%3A%2F%2Fgithub.com%2Ftensorflow%2Fmodels%2Fblob%2Fmaster%2Fresearch%2Fobject_detection%2Fg3doc%2Ftf2_detection_zoo.md&v=2yQqg_mXuPQ
# tf.config.set_visible_devices([], 'GPU')
# tf.debugging.set_log_device_placement(True)

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)

tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

#Even faster
# modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz"

#Faster but less accurate model
modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8.tar.gz"

#Slower but more accurate model
# modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_inception_resnet_v2_1024x1024_coco17_tpu-8.tar.gz"

classFile = "development\models\exported-models\pretrainedModels\coco.names"
imagePath = "test\\test.mp4"

detector = Detector()

detector.readClasses(classFile)

detector.downLoadModel(modelURL)

detector.loadModel()

image = detector.predictImage(imagePath, threshold = 0.5)

cv2.imwrite("output/" + detector.modelName + ".jpg", image)
cv2.imshow("Result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()