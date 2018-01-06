import rospy
import sys
import argparse
import base64
import json

import numpy as np
import tensorflow as tf
from PIL import Image
from keras.models import model_from_yaml
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
import cv2 

activation_relu = 'relu'

# Check if this can be moved to the Class
def create_model(hd5Path):
    model = Sequential()

    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(64, 64, 3)))

    # starts with five convolutional and maxpooling layers
    model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2, 2)))
    model.add(Activation(activation_relu))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2, 2)))
    model.add(Activation(activation_relu))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2, 2)))
    model.add(Activation(activation_relu))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
    model.add(Activation(activation_relu))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
    model.add(Activation(activation_relu))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Flatten())

    # Next, five fully connected layers
    model.add(Dense(1164))
    model.add(Activation(activation_relu))

    model.add(Dense(100))
    model.add(Activation(activation_relu))

    model.add(Dense(50))
    model.add(Activation(activation_relu))

    model.add(Dense(10))
    model.add(Activation(activation_relu))

    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')

    model.load_weights(hd5Path)

    return model


# Create model and load weights
model = create_model("model_optic_deep.h5")
model._make_predict_function()
graph = tf.get_default_graph()


class DeepDrive:

    def __init__(self):

        rospy.init_node('behavior_cloning', anonymous=True)

        self.image_sub = rospy.Subscriber("camera/image_raw", Image, self.telemetry)
        self.bridge = CvBridge()
        self.vel_pub = rospy.Publisher("/cmd_vel", Twist,queue_size=10)
        self.compressed = False
        self.prev_image = None
        self.last = []

    def crop(self, image, top_cropping_percent):
        assert 0 <= top_cropping_percent < 1.0, 'top_cropping_percent should be between zero and one'
        percent = int(np.ceil(image.shape[0] * top_cropping_percent))
        return image[percent:, :, :]

    def process(self, img):
        prev_image = self.prev_image if self.prev_image is not None else img
        self.prev_image = img
        prev = cv2.cvtColor(prev_image, cv2.COLOR_RGB2GRAY)
        next = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        self.last.append(flow)

        if len(self.last) > 4:
            self.last.pop(0)

        weights = [1, 1, 2, 2]
        last = list(self.last)
        for x in range(len(last)):
            last[x] = last[x] * weights[x]

        avg_flow = sum(self.last) / len(self.last)
        mag, ang = cv2.cartToPolar(avg_flow[..., 0], avg_flow[..., 1])

        hsv = np.zeros_like(prev_image)
        hsv[..., 1] = 255
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return rgb

    def telemetry(self, data):
        if self.compressed:
            cv_img = self.bridge.compressed_imgmsg_to_cv2(data, desired_encoding="bgr8")
        else:
            cv_img = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")

        cv_img = self.process(cv_img)
        image_array = np.asarray(cv_img)

        image_array = helper.crop(image_array, 0.25, 0.1)
        image_array = helper.resize(image_array, new_dim=(64, 64))

        transformed_image_array = image_array[None, :, :, :]

        with graph.as_default():
            steering_angle = float(model.predict(transformed_image_array, batch_size=1))

        # self.publish_command(steering_angle)

        throttle = 0.3
        vel_msg = Twist()
        vel_msg.linear.x = throttle
        vel_msg.linear.y = 0
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = steering_angle
	print(vel_msg.angular.z, steering_angle)
        # To Do: 
        # Add a PID control here
        self.vel_pub.publish(vel_msg)


    def __del__(self):
	vel_msg = Twist()
        self.vel_pub.publish(vel_msg)


def main(args):
    
    # Intialize DeepDrive class
    behavior_cloning = DeepDrive()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        print("Shutting Down")

if __name__ == "__main__":
    main(sys.argv)
