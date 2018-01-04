#!/usr/bin/env python
import rosbag
import sys
import cv2
import os
import roslib
import rosbag
import rospy
import sys
import numpy as np
import csv

from geometry_msgs.msg import Twist

STEERING_FILE = "steering.csv"

def extract(bagfile, topic, steering_file):
	bag = rosbag.Bag(bagfile)

	with open(steering_file, 'w') as f:
		for topic, msg, t in bag.read_messages(topics=[topic]):
			f.write(str(msg.angular.z))
			f.write("\n")
	bag.close()

if __name__ == '__main__':
	# if len(sys.argv) <= 2:
	# 	print 'python extract_steering_angles.py <bag file> <topic>'
	# sys.exit()
	# bagfile = sys.argv[1]
	bagfile = "track.bag"
	topic = "/cmd_vel"
	extract(bagfile, topic, STEERING_FILE)
