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
STEERING_FILE = "steering.ts"

def getTime(msg):
	return (msg.header.stamp.secs*1000) + (msg.header.stamp.nsecs/1000000)

def extractImages(bagfile, topicname, outputdir, camera_file, compressed):
	bag = rosbag.Bag(bagfile)
	bridge = CvBridge() 

	index = 1
	timestamps = []
	steering_angle = []
	for topic, msg, t in bag.read_messages(topics = [topicname]):
		ti = (getTime(msg)/50) * 50 
		
		if compressed: 
			cv_img = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
		else: 
			cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8") 

		cv2.imwrite(os.path.join(outputdir, str(index) + ".jpg"), cv_img)
		timestamps.append(ti)
		index += 1

	with open(STEERING_FILE, "r") as f: 
		steering_angle = f.read().splitlines()


	with open(camera_file, "w") as f:
		index = 1
		for ts, angle in zip(timestamps,steering_angle): 
			f.write("data/IMG/" + str(index) + ".jpg" + "," + str(ts) + "," + str(angle))
			f.write("\n")
			index += 1
	bag.close()


if __name__ == "__main__":
	if len(sys.argv) <= 1:
		print "python extarct_images.py <bagfile path> <topicname> <outputdir to persist imaes> <compressed>"
		sys.exit() 

	bagfile = sys.argv[1]
	topicname = sys.argv[2]
	outputdir = sys.argv[3]
	os.makedirs(outputdir)
	compressed = False 
	if len(sys.argv) == 5 and sys.argv[4] == 'True':
		compressed = True 
	extractImages(bagfile, topicname, outputdir, CAMERA_TIME_FILE, compressed)