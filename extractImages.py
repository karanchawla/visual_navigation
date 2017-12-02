# Created by: Karan Chawla
# Date: 11/28/17
#
# Script Description:
# 	Script is designed to take various commandline arguments making a very simple
# 	and user-friendly method to extract data from rosbag file. 
#
# Current Recommended Usage: (in terminal)
# 	python extract_images.py <bagfile path> <image_topic> <steering_angle_topic> <outputdir to persist imaes> <compressed>

#!/usr/bin/env python
import cv2
import os 
import roslib 
import rosbag 
import rospy 
import sys 
import numpy as np 
import csv 

from cv_bridge import CvBridge, CvBridgeError 
from sensor_msgs.msg import Image 
from std_msgs.msg import String 

CAMERA_TIME_FILE = "camera.csv"
STEERING_FILE = "steering.csv"

# Class for functions used for extracting required information from the rosbag
class ExtractFilesRosBag:

	# Constructor
	def __init__(self):
		self.timestamps = []
		self.steering_angles = []

	# Funtion to convert rosbag time
	def getTime(self, msg):
		"""
    	Get time data from the rosbag file.

	    Args:
    	    msg: msg for which time data needs to be extracted.

	    Returns:
    	    Returns time value for the given message.
		"""
		return (msg.header.stamp.secs*1000) + (msg.header.stamp.nsecs/1000000) 

	# Function to extract images from a bagfile, save them in a directory and 
	# create a csv file with image, timestamp and corresponding steering angle
	def extractImages(self, bagfile, topicname, outputdir, camera_file, compressed): 
		"""
    	Extracts images from the rosbag and saves it to the given outputdir.

	    Args:
    	    bagfile: The bagfile from which the images need to be extracted.
       		
       		topicname: The topicname on which the camera images are published.
			
			outputdir: The output directory in which you want the images to be saved. 
			This directory is created at runtime so only the path to the output directory 
			needs to be specified. 

			camera_file: The csv file to which the image names will be saved. 

			compressed: Boolean value to specify if extracting from a compressed image topic.
		"""
		bag = rosbag.Bag(bagfile)
		bridge = CvBridge()
		index = 1

		for topic, msg, t in bag.read_messages(topics = [topicname]):
			ti = (getTime(msg)/50) * 50 
		
			if compressed: 
				cv_img = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
			else: 
				cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8") 

			cv2.imwrite(os.path.join(outputdir, str(index) + ".jpg"), cv_img)
			self.timestamps.append(ti)
			index += 1

		with open(STEERING_FILE, "r") as f: 
			self.steering_angle = f.read().splitlines()

		with open(camera_file, "w") as f:
			index = 1
			for ts, angle in zip(self.timestamps, self.steering_angle): 
				f.write("data/IMG/" + str(index) + ".jpg" + "," + str(ts) + "," + str(angle))
				f.write("\n")
				index += 1
		bag.close() 


	# Extract steering angles from the bagfile
	def extractAngles(self, bagfile, topic, steering_file):
		"""
   		Generates points along the circumference of a circle of given radius.

	    Args:
    	    bagfile: The bagfile from which the images need to be extracted.

        	topic: The topicname on which the camera images are published.
        	
        	steering_file: The csv file to which the image names will be saved. 
		"""
		bag = rosbag.Bag(bagfile)

		with open(steering_file, 'w') as f:
			for topic, msg, t in bag.read_messages(topics=[topic]): 
				f.write(str(msg.angular.z))
				f.write("\n")
		bag.close() 


# Running the program
if __name__ == "__main__":
	if len(sys.argv) <= 1:
		print("python extract_images.py <bagfile path> <image_topic> <steering_angle_topic> <outputdir to persist imaes> <compressed>")
		sys.exit() 

	bagfile = sys.argv[1]
	image_topic = sys.argv[2]
	steering_angle_topic = sys.argv[3]
	outputdir = sys.argv[4]
	os.makedirs(outputdir)
	compressed = False 
	if len(sys.argv) == 6 and sys.argv[5] == 'True':
		compressed = True 

	extract_files = ExtractFilesRosBag()
	
	extract_files.extractAngles(bagfile, steering_angle_topic, STEERING_FILE) #It is important to extract angles before the images

	extract_images.extractImages(bagfile, image_topic, outputdir, CAMERA_TIME_FILE, compressed)