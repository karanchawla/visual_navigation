import argparse
import base64
import json

import numpy as np
import tensorflow as tf
from PIL import Image
from keras.models import model_from_json

#ROS messages
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError 
from sensor_msgs.msg import Image
from std_msgs.msg import String

import helper

model = None
prev_image_array = None

steering_pub = rospy.Publisher('/cmd_vel', Twist)

class DeepDrive: 

    def __init__(self):
        self.image_sub = rospy.Subscriber("camera/image_raw", Image, self.telemetry)
        self.bridge = CvBridge()
        self.vel_pub = rospy.Publisher("/cmd_vel", Twist)
        self.compressed = False

    def crop(self, image, top_cropping_percent):
        assert 0 <= top_cropping_percent < 1.0, 'top_cropping_percent should be between zero and one'
        percent = int(np.ceil(image.shape[0] * top_cropping_percent))
        return image[percent:, :, :]

    def telemetry(self, data):

        if self.compressed: 
            cv_img = bridge.compressed_imgmsg_to_cv2(data, desired_encoding="bgr8")
        else: 
            cv_img = bridge.imgmsg_to_cv2(data, desired_encoding="bgr8") 
    
        image_array = np.asarray(cv_img)

        image_array = helper.crop(image_array, 0.35, 0.1)
        image_array = helper.resize(image_array, new_dim=(64, 64))

        transformed_image_array = image_array[None, :, :, :]

        steering_angle = float(model.predict(transformed_image_array, batch_size=1))

        # The driving model currently just outputs a constant throttle. 
        # TO DO: Implement a PID controller here.
        throttle = 0.3
        
        vel_msg = Twist()
        vel_msg.linear.x = throttle
        vel_msg.linear.y = 0
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = steering_angle

        vel_pub.publish(vel_msg)

def main(args):
    behavior_cloning = DeepDrive()
    rospy.init_node('behavior_cloning', anonymous=True)
    model = #Add model path here
    with open(model, 'r') as jfile:
        model = model_from_json(json.load(jfile))

    model.compile("adam", "mse")
    weights_file = model.replace('json', 'h5')
    model.load_weights(weights_file)

    try: 
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting Down")


if __name__ == "__main__":
    main(sys.argv)