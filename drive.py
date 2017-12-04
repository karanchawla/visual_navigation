import rospy
import sys
import argparse
import base64
import json

import numpy as np
import tensorflow as tf
from PIL import Image
from keras.models import model_from_json
from keras.layers import Dense, Flatten, Lambda, Activation, MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential
from keras.optimizers import Adam

#ROS messages
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError 
from sensor_msgs.msg import Image
from std_msgs.msg import String

import helper

model = None

class DeepDrive: 

    def __init__(self, model_):
        self.image_sub = rospy.Subscriber("camera/image_raw", Image, self.telemetry)
        self.bridge = CvBridge()
        self.vel_pub = rospy.Publisher("/cmd_vel", Twist,queue_size=10)
        self.compressed = False
        self.model = model_

    def crop(self, image, top_cropping_percent):
        assert 0 <= top_cropping_percent < 1.0, 'top_cropping_percent should be between zero and one'
        percent = int(np.ceil(image.shape[0] * top_cropping_percent))
        return image[percent:, :, :]

    def telemetry(self, data):

        if self.compressed: 
            cv_img = self.bridge.compressed_imgmsg_to_cv2(data, desired_encoding="bgr8")
        else: 
            cv_img = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8") 
    
        image_array = np.asarray(cv_img)

        image_array = helper.crop(image_array, 0.35, 0.1)
        image_array = helper.resize(image_array, new_dim=(64, 64))

        transformed_image_array = image_array[None, :, :, :]

        steering_angle = float(self.model.predict(transformed_image_array))

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
    model_path = "model.json"
    hd5_path = "model.h5"
    # model_file = open(model_path,'r')
    # loaded_model_json = model_file.read()
    # model_file.close()

    # loaded_model = model_from_json(loaded_model_json)

    yaml_file = open('model.yaml', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model = model_from_yaml(loaded_model_yaml)

    loaded_model.compile(optimizer='adam', loss='mse')
    
    weights_file = model_path.replace('json', 'h5')
    loaded_model.load_weights(hd5_path)

    behavior_cloning = DeepDrive(loaded_model)
    rospy.init_node('behavior_cloning', anonymous=True)
    
    
    try: 
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting Down")


if __name__ == "__main__":
    main(sys.argv)