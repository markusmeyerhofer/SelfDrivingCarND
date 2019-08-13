from styx_msgs.msg import TrafficLight
import tensorflow as tf
import cv2, rospkg, rospy
from tl_classifier_trainer import TLClassifier_Trainer

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        # helper vars
        self.i = 0
        self.debug = True
        self.capture_images = False

        rospack = rospkg.RosPack()
        modelMetaFile = str(rospack.get_path('tl_detector'))+'/light_classification/model.meta'
        modelCheckpointFile = str(rospack.get_path('tl_detector'))+'/light_classification/.'

        self.x = tf.placeholder(tf.float32, (None, 60, 40, 3))
        self.y = tf.placeholder(tf.int32, (None))
        self.trainer = TLClassifier_Trainer()
        self.logits = self.trainer.LeNet(tf.cast(self.x, tf.float32))
        self.saver = tf.train.Saver()
        self.init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.saver = tf.train.import_meta_graph(modelMetaFile)
        self.saver.restore(self.sess, tf.train.latest_checkpoint(modelCheckpointFile))
        
    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        res = None
        res = cv2.resize(image ,None,fx=0.2, fy=0.2, interpolation = cv2.INTER_CUBIC)
        image = res.reshape(1, 60, 40, 3)
        assert image.shape == ((1, 60, 40, 3))
        prediction = self.sess.run(self.logits, feed_dict={self.x: image})
        classification = np.argmax(prediction)

        choices = {1: TrafficLight.GREEN, 2: TrafficLight.YELLOW, 3: TrafficLight.RED}
        result = choices.get(classification, TrafficLight.UNKNOWN)
        
        if self.debug:
            rospy.loginfo('[TL Classifier] ' + result + ' detected')
        if self.capture_images:
            imgPath = str(rospack.get_path('tl_detector'))+'/light_classification/pics/out/'
            cv2.imwrite(imgPath+str(self.i)+'.jpg', image)
            self.i += 1
        
        return result
